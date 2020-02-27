#include <iostream>
#include <cmath>
#define PI 3.14159265
using namespace std;
// GPS: latitude, longitude
//  theta stands for latitude in degree, L stands for longitude in degree
double bearing_angle(double L_A, double theta_A, double L_B, double theta_B) {
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
  double theta_A = 39.099912;
  double L_A = -94.581213;
  double theta_B = 38.627089;
  double L_B = -90.200203;
  double output = bearing_angle(L_A, theta_A, L_B, theta_B);
  cout << "bearing angle is " << output << " degree" << endl;
  return 0;
}
