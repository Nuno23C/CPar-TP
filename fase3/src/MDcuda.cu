/*
 MD.c - a simple molecular dynamics program for simulating real gas properties of Lennard-Jones particles.

 Copyright (C) 2016  Jonathan J. Foley IV, Chelsea Sweet, Oyewumi Akinfenwa

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.

 Electronic Contact:  foleyj10@wpunj.edu
 Mail Contact:   Prof. Jonathan Foley
 Department of Chemistry, William Paterson University
 300 Pompton Road
 Wayne NJ 07470

 */
#include "MDcuda.h"
#include <cuda_runtime.h>

#define NUM_THREADS_PER_BLOCK 256

const int N=5000;
const int NUM_BLOCKS = (N + NUM_THREADS_PER_BLOCK - 1)/ NUM_THREADS_PER_BLOCK;


//  Lennard-Jones parameters in natural units!
double sigma = 1.;
double epsilon = 1.;
double m = 1.;
double kB = 1.;
double Pot=0;

double NA = 6.022140857e23;
double kBSI = 1.38064852e-23;  // m^2*kg/(s^2*K)

//  Size of box, which will be specified in natural units
double L;

//  Initial Temperature in Natural Units
double Tinit;

double half_dt, three_dt;

const int MAXPART=5001;
//  Position
double r[MAXPART][3];
//  Velocity
double v[MAXPART][3];
//  Acceleration
double a[MAXPART][3];
//  Force
double F[MAXPART][3];

// atom type
char atype[10];
//  Function prototypes
//  initialize positions on simple cubic lattice, also calls function to initialize velocities
void initialize();
//  update positions and velocities using Velocity Verlet algorithm
//  print particle coordinates to file for rendering via VMD or other animation software
//  return 'instantaneous pressure'
double VelocityVerlet(double dt, int iter, FILE *fp);
//  Compute Force using F = -dV/dr
//  solve F = ma for use in Velocity Verlet
void acc_pot_aux();
//  Numerical Recipes function for generation gaussian distribution
double gaussdist();
//  Initialize velocities according to user-supplied initial Temperature (Tinit)
void initializeVelocities();
//  Compute mean squared velocity from particle velocities
double MeanSquaredVelocity();
//  Compute total kinetic energy from particle mass and velocities
double Kinetic();

int main()
{
    //  variable delcarations
    int i;
    double dt, Vol, Temp, Press, Pavg, Tavg, rho;
    double VolFac, TempFac, PressFac, timefac;
    double KE, PE, mvs, gc, Z;
    char prefix[1000], tfn[1000], ofn[1000], afn[1000];
    FILE *tfp, *ofp, *afp;


    printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("                  WELCOME TO WILLY P CHEM MD!\n");
    printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("\n  ENTER A TITLE FOR YOUR CALCULATION!\n");
    scanf("%s",prefix);
    strcpy(tfn,prefix);
    strcat(tfn,"_traj.xyz");
    strcpy(ofn,prefix);
    strcat(ofn,"_output.txt");
    strcpy(afn,prefix);
    strcat(afn,"_average.txt");

    printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("                  TITLE ENTERED AS '%s'\n",prefix);
    printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

    /*     Table of values for Argon relating natural units to SI units:
     *     These are derived from Lennard-Jones parameters from the article
     *     "Liquid argon: Monte carlo and molecular dynamics calculations"
     *     J.A. Barker , R.A. Fisher & R.O. Watts
     *     Mol. Phys., Vol. 21, 657-673 (1971)
     *
     *     mass:     6.633e-26 kg          = one natural unit of mass for argon, by definition
     *     energy:   1.96183e-21 J      = one natural unit of energy for argon, directly from L-J parameters
     *     length:   3.3605e-10  m         = one natural unit of length for argon, directly from L-J parameters
     *     volume:   3.79499-29 m^3        = one natural unit of volume for argon, by length^3
     *     time:     1.951e-12 s           = one natural unit of time for argon, by length*sqrt(mass/energy)
     ***************************************************************************************/

    //  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //  Edit these factors to be computed in terms of basic properties in natural units of
    //  the gas being simulated


    printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("  WHICH NOBLE GAS WOULD YOU LIKE TO SIMULATE? (DEFAULT IS ARGON)\n");
    printf("\n  FOR HELIUM,  TYPE 'He' THEN PRESS 'return' TO CONTINUE\n");
    printf("  FOR NEON,    TYPE 'Ne' THEN PRESS 'return' TO CONTINUE\n");
    printf("  FOR ARGON,   TYPE 'Ar' THEN PRESS 'return' TO CONTINUE\n");
    printf("  FOR KRYPTON, TYPE 'Kr' THEN PRESS 'return' TO CONTINUE\n");
    printf("  FOR XENON,   TYPE 'Xe' THEN PRESS 'return' TO CONTINUE\n");
    printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    scanf("%s",atype);

    if (strcmp(atype,"He")==0) {

        VolFac = 1.8399744000000005e-29;
        PressFac = 8152287.336171632;
        TempFac = 10.864459551225972;
        timefac = 1.7572698825166272e-12;

    }
    else if (strcmp(atype,"Ne")==0) {

        VolFac = 2.0570823999999997e-29;
        PressFac = 27223022.27659913;
        TempFac = 40.560648991243625;
        timefac = 2.1192341945685407e-12;

    }
    else if (strcmp(atype,"Ar")==0) {

        VolFac = 3.7949992920124995e-29;
        PressFac = 51695201.06691862;
        TempFac = 142.0950000000000;
        timefac = 2.09618e-12;
        //strcpy(atype,"Ar");

    }
    else if (strcmp(atype,"Kr")==0) {

        VolFac = 4.5882712000000004e-29;
        PressFac = 59935428.40275003;
        TempFac = 199.1817584391428;
        timefac = 8.051563913585078e-13;

    }
    else if (strcmp(atype,"Xe")==0) {

        VolFac = 5.4872e-29;
        PressFac = 70527773.72794868;
        TempFac = 280.30305642163006;
        timefac = 9.018957925790732e-13;

    }
    else {

        VolFac = 3.7949992920124995e-29;
        PressFac = 51695201.06691862;
        TempFac = 142.0950000000000;
        timefac = 2.09618e-12;
        strcpy(atype,"Ar");

    }
    printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("\n                     YOU ARE SIMULATING %s GAS! \n",atype);
    printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

    printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("\n  YOU WILL NOW ENTER A FEW SIMULATION PARAMETERS\n");
    printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("\n\n  ENTER THE INTIAL TEMPERATURE OF YOUR GAS IN KELVIN\n");
    scanf("%lf",&Tinit);
    // Make sure temperature is a positive number!
    if (Tinit<0.) {
        printf("\n  !!!!! ABSOLUTE TEMPERATURE MUST BE A POSITIVE NUMBER!  PLEASE TRY AGAIN WITH A POSITIVE TEMPERATURE!!!\n");
        exit(0);
    }
    // Convert initial temperature from kelvin to natural units
    Tinit /= TempFac;


    printf("\n\n  ENTER THE NUMBER DENSITY IN moles/m^3\n");
    printf("  FOR REFERENCE, NUMBER DENSITY OF AN IDEAL GAS AT STP IS ABOUT 40 moles/m^3\n");
    printf("  NUMBER DENSITY OF LIQUID ARGON AT 1 ATM AND 87 K IS ABOUT 35000 moles/m^3\n");

    scanf("%lf",&rho);

    Vol = N/(rho*NA);

    Vol /= VolFac;

    //  Limiting N to MAXPART for practical reasons
    if (N>=MAXPART) {

        printf("\n\n\n  MAXIMUM NUMBER OF PARTICLES IS %i\n\n  PLEASE ADJUST YOUR INPUT FILE ACCORDINGLY \n\n", MAXPART);
        exit(0);

    }
    //  Check to see if the volume makes sense - is it too small?
    //  Remember VDW radius of the particles is 1 natural unit of length
    //  and volume = L*L*L, so if V = N*L*L*L = N, then all the particles
    //  will be initialized with an interparticle separation equal to 2xVDW radius
    if (Vol<N) {

        printf("\n\n\n  YOUR DENSITY IS VERY HIGH!\n\n");
        printf("  THE NUMBER OF PARTICLES IS %i AND THE AVAILABLE VOLUME IS %f NATURAL UNITS\n",N,Vol);
        printf("  SIMULATIONS WITH DENSITY GREATER THAN 1 PARTCICLE/(1 Natural Unit of Volume) MAY DIVERGE\n");
        printf("  PLEASE ADJUST YOUR INPUT FILE ACCORDINGLY AND RETRY\n\n");
        exit(0);
    }
    // Vol = L*L*L;
    // Length of the box in natural units:
    L = pow(Vol,(1./3));

    //  Files that we can write different quantities to
    tfp = fopen(tfn,"w");     //  The MD trajectory, coordinates of every particle at each timestep
    ofp = fopen(ofn,"w");     //  Output of other quantities (T, P, gc, etc) at every timestep
    afp = fopen(afn,"w");    //  Average T, P, gc, etc from the simulation

    int NumTime;
    if (strcmp(atype,"He")==0) {

        // dt in natural units of time s.t. in SI it is 5 f.s. for all other gasses
        dt = 0.2e-14/timefac;
        //  We will run the simulation for NumTime timesteps.
        //  The total time will be NumTime*dt in natural units
        //  And NumTime*dt multiplied by the appropriate conversion factor for time in seconds
        NumTime=50000;
    }
    else {
        dt = 0.5e-14/timefac;
        NumTime=200;
    }

    half_dt = dt*0.5;
    three_dt = 3*dt*L*L;

    //  Put all the atoms in simple crystal lattice and give them random velocities
    //  that corresponds to the initial temperature we have specified
    initialize();

    acc_pot_aux();

    // Print number of particles to the trajectory file
    fprintf(tfp,"%i\n",N);

    //  We want to calculate the average Temperature and Pressure for the simulation
    //  The variables need to be set to zero initially
    Pavg = 0;
    Tavg = 0;


    int tenp = floor(NumTime/10);
    fprintf(ofp,"  time (s)              T(t) (K)              P(t) (Pa)           Kinetic En. (n.u.)     Potential En. (n.u.) Total En. (n.u.)\n");
    printf("  PERCENTAGE OF CALCULATION COMPLETE:\n  [");
    for (i=0; i<NumTime+1; i++) {

        //  This just prints updates on progress of the calculation for the users convenience
        if (i==tenp) printf(" 10 |");
        else if (i==2*tenp) printf(" 20 |");
        else if (i==3*tenp) printf(" 30 |");
        else if (i==4*tenp) printf(" 40 |");
        else if (i==5*tenp) printf(" 50 |");
        else if (i==6*tenp) printf(" 60 |");
        else if (i==7*tenp) printf(" 70 |");
        else if (i==8*tenp) printf(" 80 |");
        else if (i==9*tenp) printf(" 90 |");
        else if (i==10*tenp) printf(" 100 ]\n");
        fflush(stdout);


        // This updates the positions and velocities using Newton's Laws
        // Also computes the Pressure as the sum of momentum changes from wall collisions / timestep
        // which is a Kinetic Theory of gasses concept of Pressure
        Press = VelocityVerlet(dt, i+1, tfp);
        Press *= PressFac;

        //  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //  Now we would like to calculate somethings about the system:
        //  Instantaneous mean velocity squared, Temperature, Pressure
        //  Potential, and Kinetic Energy
        //  We would also like to use the IGL to try to see if we can extract the gas constant
        KE = Kinetic();
        mvs = KE / (N * 0.5);
        PE = Pot;

        // Temperature from Kinetic Theory
        Temp = m*mvs/(3*kB) * TempFac;

        // Instantaneous gas constant and compressibility - not well defined because
        // pressure may be zero in some instances because there will be zero wall collisions,
        // pressure may be very high in some instances because there will be a number of collisions
        gc = NA*Press*(Vol*VolFac)/(N*Temp);
        Z  = Press*(Vol*VolFac)/(N*kBSI*Temp);

        Tavg += Temp;
        Pavg += Press;

        fprintf(ofp,"  %8.4e  %20.8f  %20.8f %20.8f  %20.8f  %20.8f \n",i*dt*timefac,Temp,Press,KE, PE, KE+PE);
    }

    // Because we have calculated the instantaneous temperature and pressure,
    // we can take the average over the whole simulation here
    Pavg /= NumTime;
    Tavg /= NumTime;
    Z = Pavg*(Vol*VolFac)/(N*kBSI*Tavg);
    gc = NA*Pavg*(Vol*VolFac)/(N*Tavg);
    fprintf(afp,"  Total Time (s)      T (K)               P (Pa)      PV/nT (J/(mol K))         Z           V (m^3)              N\n");
    fprintf(afp," --------------   -----------        ---------------   --------------   ---------------   ------------   -----------\n");
    fprintf(afp,"  %8.4e  %15.5f       %15.5f     %10.5f       %10.5f        %10.5e         %i\n",i*dt*timefac,Tavg,Pavg,gc,Z,Vol*VolFac,N);

    printf("\n  TO ANIMATE YOUR SIMULATION, OPEN THE FILE \n  '%s' WITH VMD AFTER THE SIMULATION COMPLETES\n",tfn);
    printf("\n  TO ANALYZE INSTANTANEOUS DATA ABOUT YOUR MOLECULE, OPEN THE FILE \n  '%s' WITH YOUR FAVORITE TEXT EDITOR OR IMPORT THE DATA INTO EXCEL\n",ofn);
    printf("\n  THE FOLLOWING THERMODYNAMIC AVERAGES WILL BE COMPUTED AND WRITTEN TO THE FILE  \n  '%s':\n",afn);
    printf("\n  AVERAGE TEMPERATURE (K):                 %15.5f\n",Tavg);
    printf("\n  AVERAGE PRESSURE  (Pa):                  %15.5f\n",Pavg);
    printf("\n  PV/nT (J * mol^-1 K^-1):                 %15.5f\n",gc);
    printf("\n  PERCENT ERROR of pV/nT AND GAS CONSTANT: %15.5f\n",100*fabs(gc-8.3144598)/8.3144598);
    printf("\n  THE COMPRESSIBILITY (unitless):          %15.5f \n",Z);
    printf("\n  TOTAL VOLUME (m^3):                      %10.5e \n",Vol*VolFac);
    printf("\n  NUMBER OF PARTICLES (unitless):          %i \n", N);

    fclose(tfp);
    fclose(ofp);
    fclose(afp);

    return 0;
}


void initialize() {
    int n, p, i, j, k;
    double pos;

    // Number of atoms in each direction
    n = int(ceil(cbrt(N)));

    //  spacing between atoms along a given direction
    pos = L / n;

    //  index for number of particles assigned positions
    p = 0;
    //  initialize positions
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            for (k=0; k<n; k++) {
                if (p<N) {

                    r[p][0] = (i + 0.5)*pos;
                    r[p][1] = (j + 0.5)*pos;
                    r[p][2] = (k + 0.5)*pos;
                }
                p++;
            }
        }
    }

    // Call function to initialize velocities
    initializeVelocities();

}


//  Function to calculate the kinetic energy of the system
double Kinetic() { //Write Function here!

    double v2, kin;

    kin =0.;

    for (int i=0; i<N; i++) {

        v2 = v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2];

        kin += m*v2*0.5;
    }

    return kin;
}


__global__ void computeAccelerationsAndPotential(double (*d_r)[3], double (*d_a)[3], double* d_Pot) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = bid * blockDim.x + tid;

    __shared__ double shared_r[NUM_THREADS_PER_BLOCK][3];

    if (i < N) {

        for (int k = 0; k < 3; ++k) {
            shared_r[tid][k] = d_r[i][k];
        }

        d_Pot[i] = 0;

        double rSqd, rij[3], rSqd3, rSqd6, f;
        double temp_Pot = 0.0;
        double temp_d_a[3] = {0.0, 0.0, 0.0};

        for (int j = 0; j < N; j++) {

            if (i != j) {
                rSqd = 0;

                for (int k = 0; k < 3; k++) {
                    rij[k] = shared_r[tid][k] - d_r[j][k];
                    rSqd += rij[k] * rij[k];
                }

                rSqd3 = rSqd * rSqd * rSqd;
                rSqd6 = rSqd3 * rSqd3;

                temp_Pot += 4 * ( (1-rSqd3) / rSqd6 );

                f = 24*( (2-rSqd3) / (rSqd6*rSqd) );

                for (int k = 0; k < 3; k++) {
                    temp_d_a[k] += rij[k] * f;
                }
            }
        }

        d_Pot[i] = temp_Pot;

        for (int k = 0; k < 3; k++) {
            d_a[i][k] = temp_d_a[k];
        }
    }
}


// void acc_pot_aux() {
//     int bytes = N * 3 * sizeof(double);

//     double res_Pot[N];
//     double (*d_a)[3];
//     double (*d_r)[3];
//     double *d_Pot;

//     for (int i = 0; i < N; i++) {
//         a[i][0] = a[i][1] = a[i][2] = 0;
//     }

//     Pot = 0.0;

//     cudaMalloc((void**)&d_a, bytes);
//     cudaMalloc((void**)&d_r, bytes);
//     cudaMalloc((void**)&d_Pot, N * sizeof(double));
//     checkCUDAError("allocations");

//     // cudaMemset(d_a, 0, bytes);
//     cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
//     checkCUDAError("memSet a 0");

//     cudaMemcpy(d_r, r, bytes, cudaMemcpyHostToDevice);
//     checkCUDAError("memcopy r (HostToDevice)");

//     computeAccelerationsAndPotential<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_r, d_a, d_Pot);
//     checkCUDAError("computeAccAndPotential");

//     cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost);
//     checkCUDAError("memcopy d_a (DeviceToHost)");

//     cudaMemcpy(res_Pot, d_Pot, N * sizeof(double), cudaMemcpyDeviceToHost);

//     double total = 0.0;
//     for (int i = 0; i < N; i++)
//         total += res_Pot[i];

//     cudaFree(d_a);
//     cudaFree(d_r);
//     cudaFree(d_Pot);
//     checkCUDAError("cudaFree's");
// }

void acc_pot_aux() {
    int bytes = N * 3 * sizeof(double);

    double res_Pot[N];
    double (*d_a)[3];
    double (*d_r)[3];
    double *d_Pot;

    for (int i = 0; i < N; i++)
        a[i][0] = a[i][1] = a[i][2] = 0;

    Pot = 0.0;

    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_r, bytes);
    cudaMalloc((void**)&d_Pot, N * sizeof(double));

    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, r, bytes, cudaMemcpyHostToDevice);

    computeAccelerationsAndPotential<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(d_r, d_a, d_Pot);

    cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(res_Pot, d_Pot, N * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        Pot += res_Pot[i];

    cudaFree(d_a);
    cudaFree(d_r);
    cudaFree(d_Pot);
}


// returns sum of dv/dt*m/A (aka Pressure) from elastic collisions with walls
double VelocityVerlet(double dt, int iter, FILE *fp) {
    int i, j;
    double h_dt = dt*0.5;
    double psum = 0.;

    double* result = (double*)malloc(2*sizeof(double));
    double vij[3];

    for (i=0; i<N; i++) {
        for(int k=0;k<3;k++)
            vij[k]=a[i][k]*h_dt;

        for(int k=0;k<3;k++)
            r[i][k]+=v[i][k]*dt+vij[k]*dt;

        for(int k=0;k<3;k++)
            v[i][k]+=vij[k];

    }

    //  Update accellerations from updated positions
    acc_pot_aux();

    //  Update velocity with updated acceleration
    for (i=0; i<N; i++) {
        v[i][0] += a[i][0]*h_dt;
        v[i][1] += a[i][1]*h_dt;
        v[i][2] += a[i][2]*h_dt;
    }

    // Elastic walls
    for (i=0; i<N; i++) {
        for (j=0; j<3; j++) {
            if (r[i][j]<0.) {
                v[i][j] *=-1.; //- elastic walls
                psum += m*fabs(v[i][j]);  // contribution to pressure from "left" walls
            }
            if (r[i][j]>=L) {
                v[i][j]*=-1.;  //- elastic walls
                psum += m*fabs(v[i][j]);  // contribution to pressure from "right" walls
            }
        }
    }

    return psum/(6*L*L*h_dt);
}


void initializeVelocities() {
    int i, j;
    double vCM[3] = {0, 0, 0};
    double lambda, vSqdSum = 0.0;

    for (i=0; i<N; i++) {
        for (j=0; j<3; j++) {
            //  Pull a number from a Gaussian Distribution
            v[i][j] = gaussdist();

            vCM[j] += m*v[i][j];
        }
    }

    for (i=0; i<3; i++) vCM[i] /= N*m;

    for (i=0; i<N; i++) {
        for (j=0; j<3; j++) {
            v[i][j] -= vCM[j];

            vSqdSum += v[i][j]*v[i][j];
        }
    }

    lambda = sqrt( 3*(N-1)*Tinit/vSqdSum);

    for (i=0; i<N; i++) {
        for (j=0; j<3; j++) {
            v[i][j] *= lambda;
        }
    }
}


//  Numerical recipes Gaussian distribution number generator
double gaussdist() {
    static bool available = false;
    static double gset;
    double fac, rsq, v1, v2;
    if (!available) {
        do {
            v1 = 2.0 * rand() / double(RAND_MAX) - 1.0;
            v2 = 2.0 * rand() / double(RAND_MAX) - 1.0;
            rsq = v1 * v1 + v2 * v2;
        } while (rsq >= 1.0 || rsq == 0.0);

        fac = sqrt(-2.0 * log(rsq) / rsq);
        gset = v1 * fac;
        available = true;

        return v2*fac;
    } else {

        available = false;
        return gset;

    }
}
