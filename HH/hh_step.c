/**********************************************************************************

 Bern, 23/6/2002, (c) Michele Giugliano, Ph.D. 

 The present software aims at simulating a single conductance-based HH-like neuron,
 incorporating only a fast-inactivating sodium conductance and a delayed-rectifier
 potassium conductance, upon deterministic/stochastic current injection.

 This software is part of the Matlab-based interactive real-time computer simulation of a single
 conductance-based single-compartment biophysical model neuron under synaptic
 deterministic/stochastic stimulation (Hodgkin-Huxley-like model neuron).

 Version 1.0, Bern, 23/6/2002 - (c) 2002, Michele Giugliano, Ph.D., Physiologisches Institut, 
 Universitaet Bern, Switzerland.
 email: mgiugliano@gmail.com            url: http://www.giugliano.info

  * Windows version - compiling instructions: 
 *
 * <path>/Matlab/bin/win32/mex.bat -O hh_step.c
 *
 * (this include a call to the Microsoft Visual C++ environment compiler, and it will generate
 * a .dll output file, to be placed in the same directory of the 'fake' corresponding .m file)
 * 

 *
 *
 *
 *

 * This is a MEX-file for MATLAB.

 Demo computer simulation of a single conductance based model neuron with 
 stochastic/deterministic current injection and on-the-fly estimation of the mean firing rate
 and on the coefficient of variation of the interspike-interval distribution.
 This software takes advantage of the MATLAB visualization power, the easy and quick definition
 of a graphical user interface (GUI) and the interfacing with a 'mex' compiled c-source. It was
 developed for educational purpouses and to support with live-demos computer simulations, an 
 invited oral presentation given by the author to a conference.
 It might also be conveniently employed as a development example for further exploring the
 advantages of MATLAB GUIs and MEX interfaces. All the source codes are included in the package.

 The biophysical model simulated in the current demo software, incorporates two active membrane
 currents, following the Hodgkin-Huxley-like kinetic equations: the fast-inactivating sodium
 inward and the delayed-rectifier potassium outward currents. Together with those voltage-
 dependent currents, the total membrane current includes a passive leakage current and a
 stochastic current, representing the overall incoming afferent excitatory and inhibitory 
 synaptic current due to a large population of excitatory and inhibitory background presynaptic
 neurons, whose electrophysiological activity is not affected by the (postsynaptic) firing of the 
 simulated neuron. The simulated synaptic current result from the superposition of incoming 
 presynaptic spikes, arriving asynchronously and being modelled by instantaneous current (i.e. 
 Dirac's Delta currents). As a result, the overal resulting synaptic current is a stochastic 
 delta-correlated gaussian process whose infinitesimal moments ('mean' and 'sigma') can be 
 selected arbitrarily by the user, acting on the GUI controls.

 Please report bugs and ask for literature pointers and other details to: mgiugliano@gmail.com
 For further demos, scripts and software developed, please have a look at: www.giugliano.info .

 This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
**********************************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define SIGMA(x,theta, sigma)   1./(1.+exp(-((x)-(theta))/(sigma)))

#include "mex.h"         // Standard include library for the MEX-files.
#include "matrix.h"      // Standard include library for accessing MATLAB data struct.
static long rand49_idum=-77531;
double C   = 1.;         // [uF/cm^2] Membrane specific capacitance.
double V   = -69.965202; // [mV]      Membrane voltage.
double dt  = 0.001;      // [ms]      Integration time step (forward Euler method).

double m = 0.;           // State variable for sodium current activation.
double h = 0.;           // State variable for sodium current inactivation.
double n = 0.;           // State variable for potassium current activation.

double mean = 0.;        // Mean value of the injected current. [units are tricky here]
double stdv = 0.;        // Std dev of the injected current. [units are tricky here]

double Ina, Ik, Ileak, Iext;   // Membrane current densities for Na, K, Leak and Ext.

double hinf, tauh, thetah, sigmah, thetaht, sigmaht;  // Kinetics of inactivation (Na).
double minf, taum, thetam, sigmam;                    // Kinetics of activation (Na).
double ninf, taun, thetan, sigman, thetant, sigmant;  // Kinetics of activation (K).

double gna = 24.;      // [mS/cm^2] Specific conductance for sodium current.
double gk  = 3.;       // [mS/cm^2] Specific conductance for potassium current.
double gleak=0.25;     // [mS/cm^2] Specific conductance for leak current.

double Ena   = 55.;    // [mV] Sodium current reversal potential.
double Ek    = -90.;   // [mV] Potassium current reversal potential. 
double Eleak = -70.;   // [mV] Leakage current reversal potential.

double thetah = -53.;  // [mV] Kinetic parameter.
double sigmah = -7.;   // [mV] Kinetic parameter.
double thetaht= -40.5; // [mV] Kinetic parameter.
double sigmaht= -6.;   // [mV] Kinetic parameter.
 
double thetam=  -30.;   // [mV] Kinetic parameter.
double sigmam=  9.5;    // [mV] Kinetic parameter.
 
double thetan=  -30.;   // [mV] Kinetic parameter.
double sigman=  10.;    // [mV] Kinetic parameter.
double thetant= -27.;   // [mV] Kinetic parameter.
double sigmant= -15.;   // [mV] Kinetic parameter.

#define MM 714025
#define IA 1366
#define IC 150889

float drand49()
{
	static long iy,ir[98];
	static int iff=0;
	int j;
	
	if (rand49_idum < 0 || iff == 0) {
		iff=1;
		if((rand49_idum=(IC-rand49_idum) % MM)<0)
			rand49_idum=(-rand49_idum);
		for (j=1;j<=97;j++) {
			rand49_idum=(IA*(rand49_idum)+IC) % MM;
			ir[j]=(rand49_idum);
		}
		rand49_idum=(IA*(rand49_idum)+IC) % MM;
		iy=(rand49_idum);
	}
	j=1 + 97.0*iy/MM;
	if (j > 97 || j < 1) printf("RAN2: This cannot happen.");
	iy=ir[j];
	rand49_idum=(IA*(rand49_idum)+IC) % MM;
	ir[j]=(rand49_idum);
	return (float) iy/MM;
}

float srand49(ssseed)
long ssseed;
{
	rand49_idum=(-ssseed);
	return drand49();
}


long mysrand49(ssseed)
long ssseed;
{
	long temp;
	temp = rand49_idum;
	rand49_idum = (-ssseed);
	return temp;
}


#undef MM
#undef IA
#undef IC


float gauss()
{
	static int iset=0;
	static float gset;
	float fac,r,v1,v2;
	
	if  (iset == 0) {
		do {
			v1=2.0*drand49()-1.0;
			v2=2.0*drand49()-1.0;
			r=v1*v1+v2*v2;
		} while (r >= 1.0);
		fac=sqrt(-2.0*log(r)/r);
		gset=v1*fac;
		iset=1;
		return v2*fac;
	} else {
		iset=0;
		return gset;
	}
}

void simulate_step()
{
	minf = SIGMA(V,thetam,sigmam);
	hinf = SIGMA(V,thetah,sigmah);
	ninf = SIGMA(V,thetan,sigman);
	tauh = 0.37 + 2.78 * SIGMA(V,thetaht,sigmaht);
	taun = 0.37 + 1.85 * SIGMA(V,thetant,sigmant);
	
	Ina   = gna   * (minf*minf*minf)*h * (V - Ena);// The sodium current is updated.
	Ik    = gk    * (n*n*n*n)          * (V - Ek); // The potassium current is updated.
	Ileak = gleak * (V - Eleak);                   // The leak current is updated.
	Iext  = mean*dt + sqrt(dt)*stdv*gauss();       // The ext current is updated.
	
	V   += (dt/C) * (- Ina - Ik - Ileak) + (1./C) * Iext;
	h   += (dt/tauh) * (hinf - h);  // Please note: 'm' has been set to its equilibrium value.
	n   += (dt/taun) * (ninf - n);
	return;
} // end simulate_step()


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
	V    = *(mxGetPr(prhs[0]));
	h    = *(mxGetPr(prhs[1]));
	n    = *(mxGetPr(prhs[2]));
	mean = *(mxGetPr(prhs[3]));
	stdv = *(mxGetPr(prhs[4]));
	dt   = *(mxGetPr(prhs[5]));
	rand49_idum = (long) *(mxGetPr(prhs[6]));
	
	simulate_step();
	
	plhs[0]   =  mxCreateDoubleScalar(V);
	plhs[1]   =  mxCreateDoubleScalar(h);
	plhs[2]   =  mxCreateDoubleScalar(n); 
	plhs[3]   =  mxCreateDoubleScalar((double) -rand49_idum); 
} // end mexFunction()



