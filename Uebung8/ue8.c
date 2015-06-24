# include <stdlib.h>
# include <stdio.h>
# include <time.h>
# include <math.h>
# include <omp.h>

double __random(int *seed)
// Pseudo random number generator
// Parameters:
//   Input/Output: int *seed, a seed for the random number generator
{
	int tmp;
	double result;

	tmp = *seed / 127773;

	*seed = 16807 * (*seed - tmp * 127773) - tmp * 2836;

	if(*seed < 0)
	{
		*seed = *seed + 2147483647;
	}

	result = (double)(*seed) * 4.656612875E-10;

	return result;
}

void __initialize(unsigned int particles, unsigned int dimensions, double boxSize[], double positions[], double velocities[], double accelerations[], int *seed)
// Initializes the positions, velocities and accelerations
// Parameters:
//   Input: unsigned int particles, the number of particles
//   Input: unsigned int dimensions, the number of spatial dimensions
//   Input: double boxSize[], maximum position of particles in each dimension (Length: dimensions)
//   Output: double positions[], the position of each particle (Length: particles * dimensions)
//   Output: double velocities[], the velocity of each particle (Length: particles * dimensions)
//   Output: dobule accelerations[], the acceleration of each particle (Length: particles * dimensions)
//   Input/Output: int *seed, a seed for the random number generator
{
	// Give the particles pseudo random positions within the box
	for(unsigned int dimension = 0; dimension < dimensions; dimension++)
	for(unsigned int particle = 0; particle < particles; particle++)
	{
		positions[dimension + particle * dimensions] = boxSize[dimension] * __random(seed);
	}


	for(unsigned int particle = 0; particle < particles; particle++)
	for(unsigned int dimension = 0; dimension < dimensions; dimension++)
	{
		velocities[dimension + particle * dimensions] = 0.0;
	}

	for(unsigned int particle = 0; particle < particles; particle++)
	for(unsigned int dimension = 0; dimension < dimensions; dimension++)
	{
		accelerations[dimension + particle * dimensions] = 0.0;
	}

	return;
}

double __distance(unsigned int dimensions, double point1[], double point2[], double displacement[])
// Computes the displacement (and its norm) between two particles
// Parameters:
//   Input: unsigned int dimensions, the number of spatial dimensions
//   Input: double point1[], the position of the first particle (Length: dimensions)
//   Input: double point2[], the position of the second particle (Length: dimensions)
//   Input: double displacement[], the displacement vector of the two particles (Length: dimensions)
{
	double tmp = 0.0;

	for(unsigned int dimension = 0; dimension < dimensions; dimension++)
	{
		displacement[dimension] = point1[dimension] - point2[dimension];
		tmp = tmp + displacement[dimension] * displacement[dimension];
	}

	return sqrt(tmp);
}

void __compute(unsigned int particle, unsigned int particles, unsigned int dimensions, double positions[], double velocities[], double mass, double forces[], double *potentialEnergy, double *kineticEnergy)
// Computes the forces and energies
// Parameters:
//   Input: unsigned int particle, the number of the current particle
//   Input: unsigned int particles, the number of particles
//   Input: unsigned int dimensions, the the number of spatial dimensions
//   Input: double positions[], the positions of each particle (Length: particles * dimensions)
//   Input: double velocities[], the velocity of each particle (Length: particles * dimensions)
//   Input: double mass, the mass of each particle
//   Output: double forces[], the force on each particle (Length: particles * dimensions)
//   Output: double *potentialEnergy, the total potential energy
//   Output: double *kineticEnegery, the total kinetic energy
{
	double PI2 = 3.141592653589793 / 2.0;
	double __kineticEnergy = 0;
	double __potentialEnergy = 0;

	double displacement[3];
	double distance;
	double distance2;

	// Compute the potential energy and forces
	for(unsigned int dimension = 0; dimension < dimensions; dimension++)
	{
		forces[dimension + particle * dimensions] = 0.0;
	}

	for(unsigned int particleIndex = 0; particleIndex < particles; particleIndex++)
	{
		if(particle != particleIndex)
		{
			distance = __distance(dimensions, positions + particle * dimensions, positions + particleIndex * dimensions, displacement);

			// Attribute half of the potential energy to particle particleIndex
			if (distance < PI2)
			{
				distance2 = distance;
			}
			else
			{
				distance2 = PI2;
			}

			__potentialEnergy = __potentialEnergy + 0.5 * pow(sin(distance2), 2);

			for(unsigned int dimension = 0; dimension < dimensions; dimension++)
			{
				forces[dimension + particle * dimensions] = forces[dimension + particle * dimensions] - displacement[dimension] * sin(2.0 * distance2) / distance;
			}
		}
	}

	// Compute the kinetic energy
	for(unsigned int dimension = 0; dimension < dimensions; dimension++)
	{
		__kineticEnergy = __kineticEnergy + velocities[dimension + particle * dimensions] * velocities[dimension + particle * dimensions];
	}

	__kineticEnergy = __kineticEnergy * 0.5 * mass;
	*potentialEnergy += __potentialEnergy;
	*kineticEnergy += __kineticEnergy;

	return;
}

void __update(unsigned int particles, unsigned int dimensions, double positions[], double velocities[], double forces[], double accelerations[], double mass, double timeStepLength)
// Updates positions, velocities and accelerations
// Parameters:
//   Input: unsigned int particles, the number of particles
//   Input: unsigned int dimensions, the the number of spatial dimensions
//   Input/Output: double positions[], the position of each particle (Length: particles * dimensions)
//   Input/Output: double velocities[], the velocity of each particle (Length: particles * dimensions)
//   Input: double forces[], the force an each particle (Length: particles * dimensions)
//   Input/Output: double accelerations[], the acceleration of each particle (Length: particles * dimensions)
//   Input: double mass, the mass of each particle
//   Input: double timeStepLength, the time step length
{
	double reverseMass = 1.0 / mass;

	for(unsigned int particle = 0; particle < particles; particle++)
	for(unsigned int dimension = 0; dimension < dimensions; dimension++)
	{
		positions[dimension + particle * dimensions] = positions[dimension + particle * dimensions] + velocities[dimension + particle * dimensions] * timeStepLength + 0.5 * accelerations[dimension + particle * dimensions] * timeStepLength * timeStepLength;
		velocities[dimension + particle * dimensions] = velocities[dimension + particle * dimensions] + 0.5 * timeStepLength * (forces[dimension + particle * dimensions] * reverseMass + accelerations[dimension + particle * dimensions]);
		accelerations[dimension + particle * dimensions] = forces[dimension + particle * dimensions] * reverseMass;
	}
	return;
}

int main(int argc, char *argv[])
// Main program for the molecular dynamics simulation
// Parameters
//   None
{
	// Number of particles
	unsigned int particles = 2000;

	// Number of spatial dimensions
	unsigned int dimensions = 3;

	// Default box dimension
	double boxDimension = 10.0;

	// Number of time steps
	unsigned int timeSteps = 800;

	// Time period between two time steps
	double timeStepLength = 0.0001;

	// Mass for each particle
	double mass = 1.0;

	// Seed for the pseudo random number generator
	int seed = 123456789;

	// References to the accelerations, forces, velocities and positions of each particle
	double *accelerations, *forces, *velocities, *positions;

	// Box dimension in all spatial demensions
	double *boxSize;

	// Total potential and kinetic energy
	double potentialEnergy = 0, kineticEnergy = 0;

	// Sum of potential and kinetic energy of step 0
	double energyStep0;

	// Variables to control the time step reporting
	unsigned int timeStepPrintIndex = 0;
	unsigned int timeStepPrintOccurrence = 10;

	// Variables for time measurement
	double startTime, endTime;

	// Number of threads
	unsigned int threads;

	// Memory allocation
	accelerations = (double *) malloc(dimensions * particles * sizeof(double));
	if (accelerations == NULL)
	{
		printf("ERROR: Memory allocation for accelerations failed!\n");
		return -1;
	}

	forces = (double *) malloc(dimensions * particles * sizeof(double));
	if (forces == NULL)
	{
		printf("ERROR: Memory allocation for forces failed!\n");
		return -1;
	}

	velocities = (double *) malloc(dimensions * particles * sizeof(double));
	if (velocities == NULL)
	{
		printf("ERROR: Memory allocation for velocities failed!\n");
		return -1;
	}

	positions = (double *) malloc(dimensions * particles * sizeof(double));
	if (positions == NULL)
	{
		printf("ERROR: Memory allocation for positions failed!\n");
		return -1;
	}

	boxSize = (double *) malloc (particles * sizeof(double));
	if (boxSize == NULL)
	{
		printf("ERROR: Memory allocation for boxSize failed!\n");
		return -1;
	}

	//Set the default dimension of the box
	for(unsigned int dimension = 0; dimension < dimensions; dimension++)
	{
		boxSize[dimension] = boxDimension;
	}

	// Standard print outs
	printf("A molecular dynamics program\n");
	printf("\n");

	printf("Number of particles in the simulation: %u\n", particles);
	printf("Number of time steps:                  %u\n", timeSteps);
	printf("Length of each time step:              %.4lf\n", timeStepLength);
	printf("\n" );

	#pragma offload target(mic) out(threads)
	{
		threads = omp_get_max_threads();
	}

	printf("Number of threads available:           %u\n", threads);
	printf("\n");

	printf("Initializing positions, velocities and accelerations ...\n");

	// Set initial positions, velocities and accelerations
	__initialize(particles, dimensions, boxSize, positions, velocities, accelerations, &seed);

	// Compute the forces and energies
	printf("Computing initial forces and energies ...\n");

	for(unsigned int particle = 0; particle < particles; particle++)
	{
		__compute (particle, particles, dimensions, positions, velocities, mass, forces, &potentialEnergy, &kineticEnergy);
	}

	energyStep0 = potentialEnergy + kineticEnergy;

	// This is the main time stepping loop:
	// Compute forces and energies,
	// Update positions, velocities, accelerations


	printf("Start main computation ...\n");
	printf("\n");
	printf("At each step, the potential and kinetic energies are reported.\n");
	printf("The sum of these energies should be a constant.\n");
	printf("As an accuracy check, we also print the relative error in the total energy.\n");
	printf("\n");
	printf("      Step  Potential       Kinetic     (P+K-E0)/E0\n" );
	printf("            Energy P        Energy K    Relative Energy Error\n" );
	printf("\n");

	// Time step 0
	printf("  %8d  %010.6lf  %10.6lf  % 13.6e\n", 0, potentialEnergy, kineticEnergy, (potentialEnergy + kineticEnergy - energyStep0) / energyStep0);
	timeStepPrintIndex = timeStepPrintIndex + timeSteps / timeStepPrintOccurrence;

	startTime = omp_get_wtime();

	for (unsigned int timeStep = 1; timeStep <= timeSteps; timeStep++)
	{
		potentialEnergy = 0;
		kineticEnergy = 0;

		for(unsigned int particle = 0; particle < particles; particle++)
		{
			__compute (particle, particles, dimensions, positions, velocities, mass, forces, &potentialEnergy, &kineticEnergy);
		}

		if(timeStep == timeStepPrintIndex)
		{
			printf("  %8d  %010.6lf  %10.6lf  % 13.6e\n", timeStep, potentialEnergy, kineticEnergy, (potentialEnergy + kineticEnergy - energyStep0) / energyStep0);
			fflush(0);
			timeStepPrintIndex = timeStepPrintIndex + timeSteps / timeStepPrintOccurrence;
		}

		__update(particles, dimensions, positions, velocities, forces, accelerations, mass, timeStepLength);
	}

	endTime = omp_get_wtime();

	printf("\n");
	printf("Elapsed time for main computation:     %lf seconds\n", endTime - startTime);

	// Free allocated memory
	free(accelerations);
	free(forces);
	free(velocities);
	free(positions);
	free(boxSize);

	return 0;
}
