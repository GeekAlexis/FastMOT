#include <iostream>
#include <cmath>
#define R 6371000 // meters
#define PI 3.14159265

using namespace std;
// GPS: latitude, longitude
// theta stands for latitude in degree, L stands for longitude in degree

double distance(double theta_A, double L_A, double theta_B, double L_B) {
  L_A = L_A * PI / 180.0;
  theta_A = theta_A * PI / 180.0;
  L_B = L_B * PI / 180.0;
  theta_B = theta_B * PI / 180.0;
  double delta_theta = theta_B - theta_A;
  double delta_L = L_B - L_A;

  double a = sin(delta_theta / 2) * sin(delta_theta /2 ) + cos(theta_A) * cos(theta_B) *
        sin(delta_L / 2) * sin(delta_L / 2);
  double c = 2 * atan2(sqrt( a ), sqrt(1 - a));
  double d = R * c;
  return d;
}

int main() {
  double theta_A = 34.413738; //latitude
  double L_A = -119.841449; // longitude
  double theta_B = 34.413739;
  double L_B =   -119.841477;
  double output = distance(L_A, theta_A, L_B, theta_B);
  cout << "distance is " << output << " meters" << endl;
  return 0;
}
