#include "flight_control_sample.hpp"
#include "flight_sample.hpp"
#include <dji_telemetry.hpp>
#include <math.h>
#include <termios.h>
#include <fcntl.h>

#define STDIN_FILENO 0
#define TSCANOW 0
#define PI 3.1415926535

using namespace DJI::OSDK;
using namespace DJI::OSDK::Telemetry;

void initTermios(int echo, struct termios &oldChars, struct termios &newChars)
{
  fcntl(0, F_SETFL, O_NONBLOCK);
  tcgetattr(0, &oldChars);                 /* grab old terminal i/o settings */
  newChars = oldChars;                     /* make new settings same as old settings */
  newChars.c_lflag &= ~ICANON;             /* disable buffered i/o */
  newChars.c_lflag &= echo ? ECHO : ~ECHO; /* set echo mode */
  tcsetattr(0, TCSANOW, &newChars);        /* use these new terminal i/o settings now */
}

//  theta stands for latitude, L stands for longitude, A for current GPS, in radian, B for target GPS, in degree
double bearing_angle(double theta_A, double L_A, double theta_B, double L_B)
{
  // convert to radian
  L_B = L_B * PI / 180.0;
  theta_B = theta_B * PI / 180.0;

  double X = cos(theta_B) * sin(L_B - L_A);
  double Y = cos(theta_A) * sin(theta_B) - sin(theta_A) * cos(theta_B) * cos(L_B - L_A);
  double beta = atan2(X, Y);
  beta = beta * 180 / PI;
  return beta;
}

double distanceInBetweenEarthCoordinates(double theta_A, double L_A, double theta_B, double L_B)
{
  L_B = L_B * PI / 180.0;
  theta_B = theta_B * PI / 180.0;
  double delta_theta = theta_B - theta_A;
  double delta_L = L_B - L_A;

  double a = sin(delta_theta / 2) * sin(delta_theta / 2) + cos(theta_A) * cos(theta_B) * sin(delta_L / 2) * sin(delta_L / 2);
  double c = 2 * atan2(sqrt(a), sqrt(1 - a));
  double d = 6371000 * c;
  return d;
}

int main(int argc, char **argv)
{

  // Initialize variables
  int functionTimeout = 1;

  // Setup OSDK.
  LinuxSetup linuxEnvironment(argc, argv);
  Vehicle *vehicle = linuxEnvironment.getVehicle();
  if (vehicle == NULL)
  {
    std::cout << "Vehicle not initialized, exiting.\n";
    return -1;
  }

  // Obtain Control Authority
  vehicle->obtainCtrlAuthority(functionTimeout);

  // Display interactive prompt
  std::cout
      << "| Available commands:                                            |"
      << std::endl;
  std::cout
      << "| [a] Monitored Takeoff + Keyboard Control                       |"
      << std::endl;
  std::cout
      << "| [b] Monitored Takeoff + Waypoint Mission                       |"
      << std::endl;
  std::cout
      << "| [c] Monitored Takeoff + Waypoint Mission + Return To Home      |"
      << std::endl;

  char choice;
  char inputChar;
  std::cin >> choice;

  if (choice == 'a')
  {
    monitoredTakeoff(vehicle);
    int quit = 0;
    int angle = 0;
    int height = 1;
    double distance = 0.0;

    //BufferToggle bt;
    //bt.off();

    Telemetry::Battery battery;
    Telemetry::RTK rtk;
    Telemetry::GlobalPosition globalPosition;

    while (quit != 1)
    {
      std::cout << "Battery Percentage: " << unsigned(vehicle->broadcast->getBatteryInfo().percentage) << "\n";
      std::cout << "Battery Voltage: " << unsigned(vehicle->broadcast->getBatteryInfo().voltage) << "\n";
      angle += vehicle->broadcast->getRTKInfo().yaw; // Get the azimuth and add it to the current angle
      inputChar = std::getchar();
      if (inputChar == 'w')
      {
        if (height <= 5)
        {
          height += 1;
        }
        vehicle->control->attitudeAndVertPosCtrl(0, 0, 0, height);
        for (int i = 0; i < 5; i++)
        {
          sleep(1);
          vehicle->control->attitudeAndVertPosCtrl(0, 0, angle, height);
        }
      }
      else if (inputChar == 's')
      {
        if (height >= 2)
        {
          height -= 1;
        }
        for (int i = 0; i < 5; i++)
        {
          sleep(1);
          vehicle->control->attitudeAndVertPosCtrl(0, 0, angle, height);
        }
      }
      else if (inputChar == 'a')
      {
        angle -= 5;
        if (angle > 180)
        {
          angle -= 360;
        }
        else if (angle < -180)
        {
          angle += 360;
        }
        vehicle->control->attitudeAndVertPosCtrl(0, 0, angle, height);
      }
      else if (inputChar == 'd')
      {
        angle += 5;
        if (angle > 180)
        {
          angle -= 360;
        }
        else if (angle < -180)
        {
          angle += 360;
        }
        vehicle->control->attitudeAndVertPosCtrl(0, 0, angle, height);
      }
      else if (inputChar == 'i')
      {
        vehicle->control->attitudeAndVertPosCtrl(0, -1, angle, height);
      }
      else if (inputChar == 'k')
      {
        vehicle->control->attitudeAndVertPosCtrl(0, 1, angle, height);
      }
      else if (inputChar == 'j')
      {
        vehicle->control->attitudeAndVertPosCtrl(-1, 0, angle, height);
      }
      else if (inputChar == 'l')
      {
        vehicle->control->attitudeAndVertPosCtrl(1, 0, angle, height);
      }
      else if (inputChar == 'r')
      {
        vehicle->control->attitudeAndVertPosCtrl(0, 0, angle, height);
      }
      // Pseudo Science No Elephant
      else if (inputChar == 'x')
      {
        angle = 20;
        vehicle->control->attitudeAndVertPosCtrl(0, 0, angle, height); // Turn to the target angle
        moveByPositionOffset(vehicle, -10, 0, 0, 0);
        //vehicle->control->attitudeAndVertPosCtrl(0,5,angle,height); // Move forward by 50 meters
        moveByPositionOffset(vehicle, 0, 0, 0, -135);
        moveByPositionOffset(vehicle, 0, 0, 0, 0);
        moveByPositionOffset(vehicle, 0, 0, 0, 135);
        moveByPositionOffset(vehicle, 0, 0, 0, 0);
        moveByPositionOffset(vehicle, 0, 0, 0, 180);
        moveByPositionOffset(vehicle, 0, 0, 0, 0);
        //moveByPositionOffset(vehicle, 0, 0, 0, -5);
        //vehicle->control->attitudeAndVertPosCtrl(0,0,angle-5,height); // Turn left by 5 degrees
        //vehicle->control->attitudeAndVertPosCtrl(0,0,angle-1,height); // Turn left by 1 degree
        //vehicle->control->attitudeAndVertPosCtrl(0,0,angle-3,height); // Turn left by 3 degrees
        //vehicle->control->attitudeAndVertPosCtrl(0,0,angle+3,height); // Turn right by 3 degrees
        //vehicle->control->attitudeAndVertPosCtrl(0,0,angle+5,height); // Turn right by 5 degrees
        moveByPositionOffset(vehicle, 10, 0, 0, -45);
        //vehicle->control->attitudeAndVertPosCtrl(0,-5,angle,height); // Move backward by 5 meters because object is too close
      }
      // Pseudo Science Women Beach
      else if (inputChar == 'z')
      {
        angle = 20;
        vehicle->control->attitudeAndVertPosCtrl(0, 0, angle, height); // Turn to the target angle
        moveByPositionOffset(vehicle, -10, 0, 0, 0); // Move to the GPS coordinate
        moveByPositionOffset(vehicle, -1, 0, 0, 0); // Move forward by one meter
        moveByPositionOffset(vehicle, -1, 0, 0, 1); // Move forward by one meter and turn slightly right by one degree
        moveByPositionOffset(vehicle, -1, 0, 0, 1); // Move forward by one meter and turn slightly right by one degree
        moveByPositionOffset(vehicle, -1, 0, 0, 1); // Move forward by one meter and turn slightly right by one degree
        moveByPositionOffset(vehicle, -1, 0, 0, 1); // Move forward by one meter and turn slightly right by one degree
        moveByPositionOffset(vehicle, -1, 0, 0, 3); // Move forward by one meter and turn slightly right by three degrees
        moveByPositionOffset(vehicle, -1, 0, 0, 3); // Move forward by one meter and turn slightly right by three degrees
        moveByPositionOffset(vehicle, -1, 0, 0, 5); // Move forward by one meter and turn slightly right by five degrees
        moveByPositionOffset(vehicle, -1, 0, 0, 5); // Move forward by one meter and turn slightly right by five degrees
        moveByPositionOffset(vehicle, -1, 0, 0, 5); // Move forward by one meter and turn slightly right by five degrees
      }
      // Pseudo Science Bicycle
      else if (inputChar == 'v')
      {
        angle = 20;
        vehicle->control->attitudeAndVertPosCtrl(0, 0, angle, height); // Turn to the target angle
        moveByPositionOffset(vehicle, -10, 0, 0, 0); // Move to the GPS coordinate
        moveByPositionOffset(vehicle, -1, 1, 0, 0);
        moveByPositionOffset(vehicle, -1, 1, 0, 0);
        moveByPositionOffset(vehicle, -1, 1, 0, 0);
        moveByPositionOffset(vehicle, -1, 1, 0, 0);
        moveByPositionOffset(vehicle, -1, 1, 0, 0);
        moveByPositionOffset(vehicle, -1, 1, 0, 0);
        moveByPositionOffset(vehicle, -1, 1, 0, 0);
        moveByPositionOffset(vehicle, -1, 1, 0, 0);
        moveByPositionOffset(vehicle, -1, 1, 0, 0);
        moveByPositionOffset(vehicle, -1, 2, 0, 0);
        moveByPositionOffset(vehicle, -1, 2, 0, 0);
        moveByPositionOffset(vehicle, -1, 2, 0, 0);
        moveByPositionOffset(vehicle, -1, 2, 0, 0);
        moveByPositionOffset(vehicle, -1, 2, 0, 0);
        moveByPositionOffset(vehicle, -1, 2, 0, 0);
        moveByPositionOffset(vehicle, -1, 2, 0, 0);
        moveByPositionOffset(vehicle, -1, 0, 0, -2);
        moveByPositionOffset(vehicle, -1, 0, 0, -4);
        moveByPositionOffset(vehicle, -1, 0, 0, -5);
        moveByPositionOffset(vehicle, -1, 0, 0, -7);
        moveByPositionOffset(vehicle, -1, -1, 0, -8);
        moveByPositionOffset(vehicle, -1, -1, 0, -9);
        moveByPositionOffset(vehicle, -1, -1, 0, -10);
        moveByPositionOffset(vehicle, -1, -1, 0, -10);
        moveByPositionOffset(vehicle, -1, 0, 0, -6);
        moveByPositionOffset(vehicle, -1, 0, 0, -1);
        moveByPositionOffset(vehicle, -1, 0, 0, 3);
        moveByPositionOffset(vehicle, -1, 3, 0, 5);
        moveByPositionOffset(vehicle, -1, 3, 0, 8);
        moveByPositionOffset(vehicle, -0.5, 4, 0, 10);
        moveByPositionOffset(vehicle, -0.5, 4, 0, 10);
        moveByPositionOffset(vehicle, 0, 3, 0, 10);
      }
      // Pseudo Science Bicycle Smooth
      else if (inputChar == 'n')
      {
        angle = 20;
        vehicle->control->attitudeAndVertPosCtrl(0, 0, angle, height); // Turn to the target angle
        vehicle->control->attitudeAndVertPosCtrl(0, 2, angle, height); // Move to the GPS coordinate
        sleep(5);
        vehicle->control->attitudeAndVertPosCtrl(1, 1, angle, height);
        vehicle->control->attitudeAndVertPosCtrl(1, 1, angle, height);
        vehicle->control->attitudeAndVertPosCtrl(1, 1, angle, height);
        vehicle->control->attitudeAndVertPosCtrl(1, 1, angle, height);
        vehicle->control->attitudeAndVertPosCtrl(1, 1, angle, height);
        vehicle->control->attitudeAndVertPosCtrl(1, 1, angle, height);
        vehicle->control->attitudeAndVertPosCtrl(1, 1, angle, height);
        vehicle->control->attitudeAndVertPosCtrl(1, 1, angle, height);
        vehicle->control->attitudeAndVertPosCtrl(1, 1, angle, height);
        vehicle->control->attitudeAndVertPosCtrl(1, 1, angle, height);
        vehicle->control->attitudeAndVertPosCtrl(2, 1, angle, height);
        vehicle->control->attitudeAndVertPosCtrl(2, 1, angle, height);
        vehicle->control->attitudeAndVertPosCtrl(2, 1, angle, height);
        vehicle->control->attitudeAndVertPosCtrl(2, 1, angle, height);
        vehicle->control->attitudeAndVertPosCtrl(2, 1, angle, height);
        vehicle->control->attitudeAndVertPosCtrl(2, 1, angle, height);
        vehicle->control->attitudeAndVertPosCtrl(2, 1, angle, height);
        vehicle->control->attitudeAndVertPosCtrl(2, 1, angle, height);
        vehicle->control->attitudeAndVertPosCtrl(0, 1, angle-2, height);
        vehicle->control->attitudeAndVertPosCtrl(0, 1, angle-4, height);
        vehicle->control->attitudeAndVertPosCtrl(0, 1, angle-5, height);
        vehicle->control->attitudeAndVertPosCtrl(0, 1, angle-7, height);
        vehicle->control->attitudeAndVertPosCtrl(-1, 1, angle-8, height);
        vehicle->control->attitudeAndVertPosCtrl(-1, 1, angle-9, height);
        vehicle->control->attitudeAndVertPosCtrl(-1, 1, angle-10, height);
        vehicle->control->attitudeAndVertPosCtrl(-1, 1, angle-10, height);
        vehicle->control->attitudeAndVertPosCtrl(0, 1, angle-6, height);
        vehicle->control->attitudeAndVertPosCtrl(0, 1, angle-1, height);
        vehicle->control->attitudeAndVertPosCtrl(0, 1, angle+3, height);
        vehicle->control->attitudeAndVertPosCtrl(0, 1, angle+3, height);
        vehicle->control->attitudeAndVertPosCtrl(3, 1, angle+5, height);
        vehicle->control->attitudeAndVertPosCtrl(3, 1, angle+8, height);
        vehicle->control->attitudeAndVertPosCtrl(3, 1, angle+8, height);
        vehicle->control->attitudeAndVertPosCtrl(4, 0.5, angle+10, height);
        vehicle->control->attitudeAndVertPosCtrl(4, 0.5, angle+10, height);
        vehicle->control->attitudeAndVertPosCtrl(3, 0, angle+10, height);
        vehicle->control->attitudeAndVertPosCtrl(3, 0, angle+10, height);
        vehicle->control->attitudeAndVertPosCtrl(3, 0, angle+10, height);
      }
      else if (inputChar == 't')
      {
        double lat = 34.41508935247427;
        double log = -119.84354498675536;

        if (angle > 180)
        {
          angle -= 360;
        }
        else if (angle < -180)
        {
          angle += 360;
        }
        angle += bearing_angle(vehicle->broadcast->getGlobalPosition().latitude, vehicle->broadcast->getGlobalPosition().longitude, lat, log);
        angle -= vehicle->broadcast->getRTKInfo().yaw;
        if (angle > 180)
        {
          angle -= 360;
        }
        else if (angle < -180)
        {
          angle += 360;
        }
        vehicle->control->attitudeAndVertPosCtrl(0, 0, angle, height); // Turn to the target angle
        distance = distanceInBetweenEarthCoordinates(vehicle->broadcast->getGlobalPosition().latitude, vehicle->broadcast->getGlobalPosition().longitude, lat, log);
        std::cout << "Distance: " << distance << " Meters\n";
        while (distance >= 5)
        {
          vehicle->control->attitudeAndVertPosCtrl(0, -1, angle, height); // Move forward towards the target location with the calculated angle by one meter
          std::cout << "Bearing Angle: " << bearing_angle(vehicle->broadcast->getGlobalPosition().latitude, vehicle->broadcast->getGlobalPosition().longitude, lat, log) << "\n";
          angle += bearing_angle(vehicle->broadcast->getGlobalPosition().latitude, vehicle->broadcast->getGlobalPosition().longitude, lat, log); // Recalculate the angle
          std::cout << "RTK Angle: " << vehicle->broadcast->getRTKInfo().yaw << "\n";
          angle -= vehicle->broadcast->getQuaternion().q3 * 90;
          std::cout << "Quaternion Angle: " << vehicle->broadcast->getQuaternion().q3 << "\n";
          std::cout << "Angle After: " << angle << "\n";
          if (angle > 180)
          {
            angle -= 360;
          }
          else if (angle < -180)
          {
            angle += 360;
          }
          distance = distanceInBetweenEarthCoordinates(vehicle->broadcast->getGlobalPosition().latitude, vehicle->broadcast->getGlobalPosition().longitude, lat, log);
          std::cout << "Distance: " << distance << " Meters\n";
        }
      }
      else if (inputChar == 'g')
      {
        runWaypointMission2(vehicle, 1, functionTimeout, 34.41508935247427 * PI / 180, -119.84354498675536 * PI / 180); // Waypoint mission
      }
      else if (inputChar == 'q')
      {
        quit = 1;
        //bt.off();
      }
    }
    monitoredLanding(vehicle);
  }

  else if (inputChar == 'b')
  {
    // monitoredTakeoff(vehicle);
    runWaypointMission2(vehicle, 1, functionTimeout, 34.41508935247427 * PI / 180, -119.84354498675536 * PI / 180); // Waypoint mission
    // monitoredLanding(vehicle);
  }

  else if (inputChar == 'c')
  {
    // monitoredTakeoff(vehicle);
    runWaypointMission2(vehicle, 2, functionTimeout, 34.41508935247427 * PI / 180, -119.84354498675536 * PI / 180); // Waypoint mission
    // monitoredLanding(vehicle);
  }

  return 0;
}
