#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
#include <iomanip>
#include <sstream>
using namespace std;

// Constants
const int N = 64;                  // Número de partícules
const double rho = 0.8;            // Densitat
const double T_target = 1.0;       // Temperatura
const double dt = 0.005;           // Pas de temps
const int Steps = 1000;           // Nombre de passos de simulació
const double rcut = 3.0;           // Distància de tall
const double epsilon = 1.0;
const double sigma = 1.0;
const int frame_stride = 10;       // Cada quants passos escriure .xyz

// Dimensions de la caixa
double L;

// Vector 2D
struct Vec2 {
    double x, y;
    Vec2(double a = 0, double b = 0) : x(a), y(b) {}
    Vec2 operator+(const Vec2& v) const { return Vec2(x + v.x, y + v.y); }
    Vec2 operator-(const Vec2& v) const { return Vec2(x - v.x, y - v.y); }
    Vec2 operator*(double s) const { return Vec2(x * s, y * s); }
    Vec2& operator+=(const Vec2& v) { x += v.x; y += v.y; return *this; }
    Vec2& operator-=(const Vec2& v) { x -= v.x; y -= v.y; return *this; }
};

struct Particle {
    Vec2 pos, vel, acc;
};

vector<Particle> particles;
double last_PE = 0.0;  // Per guardar l'última energia potencial

// Aplica condicions periòdiques
Vec2 pbc(Vec2 r) {
    if (r.x >  L / 2) r.x -= L;
    if (r.x < -L / 2) r.x += L;
    if (r.y >  L / 2) r.y -= L;
    if (r.y < -L / 2) r.y += L;
    return r;
}

// Inicialitza posicions en malla regular
void init_positions() {
    int n = ceil(sqrt(N));
    double spacing = L / n;
    int count = 0;
    for (int i = 0; i < n && count < N; ++i) {
        for (int j = 0; j < n && count < N; ++j) {
            particles[count].pos = Vec2(i * spacing, j * spacing);
            count++;
        }
    }
}

// Inicialitza velocitats aleatòries i les rescale
void init_velocities() {
    double vx_sum = 0, vy_sum = 0;
    srand(time(0));
    for (auto& p : particles) {
        p.vel.x = (rand() / (double)RAND_MAX - 0.5);
        p.vel.y = (rand() / (double)RAND_MAX - 0.5);
        vx_sum += p.vel.x;
        vy_sum += p.vel.y;
    }
    vx_sum /= N;
    vy_sum /= N;
    double ke = 0;
    for (auto& p : particles) {
        p.vel.x -= vx_sum;
        p.vel.y -= vy_sum;
        ke += 0.5 * (p.vel.x * p.vel.x + p.vel.y * p.vel.y);
    }
    double scale = sqrt(N * T_target / (2 * ke));
    for (auto& p : particles) {
        p.vel.x *= scale;
        p.vel.y *= scale;
    }
}

// Calcula forces i energia potencial
double compute_forces() {
    for (auto& p : particles) p.acc = Vec2(0, 0);
    double PE = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            Vec2 rij = pbc(particles[i].pos - particles[j].pos);
            double r2 = rij.x * rij.x + rij.y * rij.y;
            if (r2 < rcut * rcut) {
                double r6 = pow(sigma * sigma / r2, 3);
                double r12 = r6 * r6;
                double f = 24 * epsilon * (2 * r12 - r6) / r2;
                Vec2 force = rij * f;
                particles[i].acc += force;
                particles[j].acc -= force;
                PE += 4 * epsilon * (r12 - r6);
            }
        }
    }
    return PE;
}

// Integrador Velocity-Verlet
double velocity_verlet() {
    double KE = 0.0;
    for (auto& p : particles) {
        p.vel += p.acc * (0.5 * dt);
        p.pos += p.vel * dt;

        if (p.pos.x < 0) p.pos.x += L;
        if (p.pos.x >= L) p.pos.x -= L;
        if (p.pos.y < 0) p.pos.y += L;
        if (p.pos.y >= L) p.pos.y -= L;
    }
    last_PE = compute_forces();
    for (auto& p : particles) {
        p.vel += p.acc * (0.5 * dt);
        KE += 0.5 * (p.vel.x * p.vel.x + p.vel.y * p.vel.y);
    }
    return KE;
}

// Escriu fitxer XYZ per visualització
void write_xyz(int step) {
    ostringstream filename;
    filename << "frames/frame" << step << ".xyz";
    ofstream xyz(filename.str());
    xyz << N << "\n";
    xyz << "Pas " << step << "\n";
    for (const auto& p : particles) {
        xyz << "Ar " << p.pos.x << " " << p.pos.y << " 0.0\n";
    }
    xyz.close();
}

int main() {
    L = sqrt(N / rho);
    particles.resize(N);
    init_positions();
    init_velocities();
    compute_forces(); // forces inicials

    // Crear fitxer CSV
    ofstream dataFile("dades.csv");
    dataFile << "Pas,ECinetica,EPotencial,ETotal,Temperatura\n";

    for (int step = 0; step < Steps; ++step) {
        double KE = velocity_verlet();
        double temp = (2 * KE) / (2.0 * N);  // kB = 1, 2D
        double PE = last_PE;
        dataFile << step << "," << KE << "," << PE << "," << (KE + PE) << "," << temp << "\n";

        if (step % 100 == 0) {
            cout << "Pas " << step << " | E = " << KE + PE << " | T = " << temp << endl;
        }

        if (step % frame_stride == 0) {
            write_xyz(step);
        }
    }

    dataFile.close();
    cout << "Simulació acabada. Dades escrites a 'dades.csv' i carpeta 'frames/'." << endl;
    return 0;
}
