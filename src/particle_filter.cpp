/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include "map.h"

#include "particle_filter.h"

using namespace std;

static int NUM_PARTICLES = 100;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	// those helped here too
	if (!is_initialized)
	{
		default_random_engine random_engine;
		num_particles = NUM_PARTICLES;

		// Gaussian normally distributed particles
		normal_distribution<double> dist_x(x, std[0]);
		normal_distribution<double> dist_y(y, std[1]);
		normal_distribution<double> dist_theta(theta, std[2]);
		particles.resize(num_particles);

		// initialize all weights to 1
		weights.resize(num_particles);
		// initialize weights 
		//fill(weights.begin(), weights.end(), 1.0);

		for (int i = 0; i < num_particles; i++)
		{
			// Create the particle object with the random x and y values
			Particle p;
			p.id = i;
			p.x = dist_x(random_engine);
			p.y = dist_y(random_engine);
			p.theta = dist_theta(random_engine);
			// initialize weight to 1.0 as indicated by the instructions
			p.weight = 1.0;
			particles.push_back(p);
			weights.push_back(1.0);
		}
		is_initialized = true;
	}
	
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	// Prediction step equations from class:
	// Xf = X0 + velocity/yaw_rate [sin(theta +  yaw_rate*delta) - sin(theta)]
	// Yf = Y0 + velocity/yaw_rate [cos(theta) - cos(theta + yaw_rate*delta)]
	// Theta0 = Theta0 + Theta*delta_t

	// random engine
	default_random_engine random_engine;

	for (int i = 0; i < num_particles; i++)
	{
		if (fabs(yaw_rate) < 0.0001)
		{
			particles[i].x +=  velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
		else 
		{
			double v_over_yaw_rate = velocity / yaw_rate;
			double yaw_rate_of_delta_t = yaw_rate * delta_t;
			particles[i].x += v_over_yaw_rate * (sin(particles[i].theta + yaw_rate_of_delta_t) - sin(particles[i].theta));
			particles[i].y += v_over_yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate_of_delta_t));
			particles[i].theta = particles[i].theta + yaw_rate_of_delta_t;
		}

		// Adding noise
		normal_distribution<double> noise_x(particles[i].x, std_pos[0]);
		normal_distribution<double> noise_y(particles[i].y, std_pos[1]);
		normal_distribution<double> noise_theta(particles[i].theta, std_pos[2]);
		
		// Add noise
		particles[i].x = noise_x(random_engine);
		particles[i].y = noise_y(random_engine);
		particles[i].theta = noise_theta(random_engine);

		//cout << "id: " << particles[i].id << " x: " << particles[i].x << " y: " << particles[i].y << " theta: " << particles[i].theta << "weights : " << particles[i].weight << "\n";
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	// Source Forum help: https://discussions.udacity.com/t/you-ran-out-of-time-when-running-with-the-simulator/269900/8

	
	for (unsigned int i = 0; i < observations.size(); i++)
	{
		const LandmarkObs observation = observations[i];
		vector <double> distances;
		for (auto prediction : predicted)
		{
			distances.push_back(dist(observation.x, observation.y, prediction.x, prediction.y));
		}
		
		vector<double>::iterator min_result = min_element(begin(distances), end(distances));
		LandmarkObs lm = predicted[distance(begin(distances), min_result)];
		observations[i].id = lm.id;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	//  x * cos(theta) - y * sin(theta) + x at delta_t
	//  x * sin(theta) + y * cos(theta) + y at delta_t

	// Thomas Anthony on Slack suggested the following workflow, and thanks to his help and credit was able to understand what was possible here:
	//1. Make list of all landmarks within sensor range of particle, call this `predicted_lm`
	//2. Convert all observations from local to global frame, call this `transformed_obs`
	//3. Perform `dataAssociation`. This will put the index of the `predicted_lm` nearest to each `transformed_obs` in the `id` field of the `transformed_obs` element.
	//4. Loop through all the `transformed_obs`. Use the saved index in the `id` to find the associated landmark and compute the gaussian.
	//5. Multiply all the gaussian values together to get total probability of particle(the weight)
	// for each particle
	//weights.clear();
	for (int i = 0; i < num_particles; i++)
	{
		particles[i].weight = 1.0;
		Particle p = particles[i];
		vector<LandmarkObs> predictedLandmarks;

		for (int l = 0; l < map_landmarks.landmark_list.size(); l++)
		{
			Map::single_landmark_s m_landmark = map_landmarks.landmark_list[l];

			// check if its in range
			double distance = dist(m_landmark.x_f, m_landmark.y_f, p.x, p.y);
			if (distance <= sensor_range * 1.5)
			{
				vector<double> particle_vector = { p.x, p.y};
				vector <double> landmark_vector = { m_landmark.x_f, m_landmark.y_f };
				double product = inner_product(particle_vector.begin(), particle_vector.end(), landmark_vector.begin(), 0.0);
				
				if ( product >= 0)
				{
					predictedLandmarks.push_back(LandmarkObs{ m_landmark.id_i, m_landmark.x_f, m_landmark.y_f });
				}
			}
		}

		// Predicted observations
		vector<LandmarkObs> transObservations;

		for (int j = 0; j < observations.size(); j++)
		{
			LandmarkObs transObservation;
			LandmarkObs obs = observations[j];

			transObservation.x = obs.x * cos(p.theta) - obs.y * sin(p.theta) + p.x;
			transObservation.y = obs.x * sin(p.theta) + obs.y * cos(p.theta) + p.y;
			transObservation.id = obs.id;
			
			transObservations.push_back(transObservation);
		}

		//cout << "Trans Observation : " << transObservations.size() << endl;
		

		dataAssociation(predictedLandmarks, transObservations);

		for (unsigned int x = 0; x < transObservations.size(); x++)
		{
			LandmarkObs observation = transObservations[x];
			LandmarkObs assocLandmark = predictedLandmarks[observation.id];

			const double sigma_x = std_landmark[0];
			const double sigma_y = std_landmark[1];

			double x_diff = pow(observation.x - assocLandmark.x, 2) /  (2 * pow(sigma_x, 2));
			double y_diff = pow(observation.y - assocLandmark.y, 2) / (2 * pow(sigma_y, 2));

			particles[i].weight *= 1 / (2 * M_PI * sigma_x * sigma_y) * exp(-(x_diff + y_diff));

			//cout << "id: " << particles[i].id << " x: " << particles[i].x << " y: " << particles[i].y << " theta: " << particles[i].theta << "weights : " << particles[i].weight << "\n";
		}
		
		weights.push_back(particles[i].weight);
		
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	//   https://stackoverflow.com/questions/40275512/how-to-generate-random-numbers-between-2-values-inclusive
	//   http://www.cplusplus.com/reference/algorithm/max_element/
	vector<Particle> resample_particles;
	random_device rd;
	mt19937 gen(rd());
	discrete_distribution<> dist(weights.begin(), weights.end());
	resample_particles.resize(num_particles);

	for (int j = 0; j < particles.size(); j++)
	{
		weights[j] = particles[j].weight;
	}
	
	for (int i = 0; i < num_particles; i++)
	{
		resample_particles.push_back(particles[dist(gen)]);
	}
	particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

void ParticleFilter::write(std::string filename)
{
	ofstream file;
	file.open(filename, std::ios::app);
	for (auto particle : particles)
	{
		file << particle.x << " " << particle.y << " " << particle.theta << endl;
	}
	file.close();
}
