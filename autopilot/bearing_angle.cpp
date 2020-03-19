#include <iostream>
#include <cmath>
#define PI 3.14159265
using namespace std;
// GPS: latitude, longitude
//  theta stands for latitude in degree, L stands for longitude in degree
// east is 0 degree; + clockwise
double bearing_angle(double theta_A, double L_A, double theta_B, double L_B) {
    // convert to radian
    L_A = L_A * PI / 180.0;
    theta_A = theta_A * PI / 180.0;
    L_B = L_B * PI / 180.0;
    theta_B = theta_B * PI / 180.0;

    double X = cos(theta_B)*sin(L_B - L_A);
    double Y = cos(theta_A)*sin(theta_B)-sin(theta_A)*cos(theta_B)*cos(L_B-L_A);
    double beta = atan2(X, Y);
    beta = beta * 180 / PI;
    // cout << "X is " << X << endl;
    // cout << "Y is " << Y << endl;
    return beta;
}

int main() {
  double theta_A = 34.413744; //latitude
  double L_A = -119.841454; // longitude
  double theta_B = 34.413376;
  double L_B =  -119.841472;
  double output = bearing_angle(L_A, theta_A, L_B, theta_B);
  cout << "bearing angle is " << output << " degree" << endl;
  return 0;
}
