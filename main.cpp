/*! @file flight-control/main.cpp
 *  @version 3.3
 *  @date Jun 05 2017
 *
 *  @brief
 *  main for Flight Control API usage in a Linux environment.
 *  Provides a number of helpful additions to core API calls,
 *  especially for position control, attitude control, takeoff,
 *  landing.
 *
 *  @Copyright (c) 2016-2017 DJI
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

/*TODO:flight_control_sample will by replace by flight_sample in the future*/
#include "flight_control_sample.hpp"
#include "flight_sample.hpp"
#include <math.h>
#include <dji_telemetry.hpp> // Added

#include <termios.h>
#include <fcntl.h>
#define STDIN_FILENO 0
#define TSCANOW 0
#define PI 3.14159265 // Added

void initTermios(int echo, struct termios &oldChars, struct termios &newChars)
{
  fcntl(0, F_SETFL, O_NONBLOCK);
  tcgetattr(0, &oldChars); /* grab old terminal i/o settings */
  newChars = oldChars; /* make new settings same as old settings */
  newChars.c_lflag &= ~ICANON; /* disable buffered i/o */
  newChars.c_lflag &= echo ? ECHO : ~ECHO; /* set echo mode */
  tcsetattr(0, TCSANOW, &newChars); /* use these new terminal i/o settings now */
}

using namespace DJI::OSDK;
using namespace DJI::OSDK::Telemetry;

//  theta stands for latitude, L stands for longitude, A for current GPS, in radian, B for target GPS, in degree
double bearing_angle(double theta_A, double L_A, double theta_B, double L_B) {
    // convert to radian
    L_B = L_B * PI / 180.0;
    theta_B = theta_B * PI / 180.0;

    double X = cos(theta_B)*sin(L_B - L_A);
    double Y = cos(theta_A)*sin(theta_B)-sin(theta_A)*cos(theta_B)*cos(L_B-L_A);
    double beta = atan2(X, Y);
    beta = beta * 180 / PI;
    return beta;
}

double degreesToRadians(double degrees) {
    return degrees * PI / 180;
}

double distanceInKmBetweenEarthCoordinates(double lat1, double lon1, double lat2, double lon2) {
    double earthRadiusKm = 6371.345;

    // convert to degree
    double lat1_degree = lat1 / PI * 180.0;
    double lon1_degree = lon1 / PI * 180.0;

    double dLat = degreesToRadians(lat2 - lat1_degree);
    double dLon = degreesToRadians(lon2 - lon1_degree);

    double a = sin(dLat/2) * sin(dLat/2) + sin(dLon/2) * sin(dLon/2) * cos(lat1) * cos(lat2);
    double c = 2 * atan2(sqrt(a), sqrt(1-a));
    return earthRadiusKm * c;
}

int main(int argc, char** argv) {

  // Initialize variables
  int functionTimeout = 1;

  // Setup OSDK.
  LinuxSetup linuxEnvironment(argc, argv);
  Vehicle* vehicle = linuxEnvironment.getVehicle();
  if (vehicle == NULL) {
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
      << "| [a] Monitored Takeoff + Landing                                |"
      << std::endl;
  std::cout
      << "| [b] Monitored Takeoff + Position Control + Landing             |"
      << std::endl;
  std::cout << "| [c] Monitored Takeoff + Position Control + Force Landing "
               "Avoid Ground  |"
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
    BufferToggle bt;
    bt.off();

    double x; 
    double y; 
    double z = atan(80 * M_PI / 180);

    Telemetry::Battery battery;
    Telemetry::RTK rtk;
    Telemetry::GlobalPosition globalPosition;

    while (quit != 1)
    {
      battery = vehicle->broadcast->getBatteryInfo();
      rtk = vehicle->broadcast->getRTKInfo();
      globalPosition = vehicle->broadcast->getGlobalPosition();
      std::cout << "Battery Percentage: " << unsigned(battery.percentage) << "\n";
      std::cout << "Battery Voltage: " << unsigned(battery.voltage) << "\n";
      angle += rtk.yaw; // Get the azimuth and add it to the current angle
      inputChar = std::getchar();
      if (inputChar == 'w')
      {
        if (height <= 5) {height += 1;}
        vehicle->control->attitudeAndVertPosCtrl(0,0,0,height);
        for (int i = 0; i < 5; i++)
        {
          sleep(1);
          vehicle->control->attitudeAndVertPosCtrl(0,0,angle,height);
        }
      }
      else if (inputChar == 's') 
      {
        if (height >= 2) {height -= 1;}
        for (int i = 0; i < 5; i++)
        {
          sleep(1);
          vehicle->control->attitudeAndVertPosCtrl(0,0,angle,height);
        }
      }
      else if (inputChar == 'a') 
      {
        angle -= 5;
        if(angle > 180) {angle -= 360;}
        else if(angle < -180) {angle += 360;}
        vehicle->control->attitudeAndVertPosCtrl(0,0,angle,height);
      }
      else if (inputChar == 'd') 
      {
        angle += 5;
        if(angle > 180) {angle -= 360;}
        else if(angle < -180) {angle += 360;}
        vehicle->control->attitudeAndVertPosCtrl(0,0,angle,height);
      }
      else if (inputChar == 'i') {vehicle->control->attitudeAndVertPosCtrl(0,-5,angle,height);}
      else if (inputChar == 'k') {vehicle->control->attitudeAndVertPosCtrl(0,5,angle,height);}
      else if (inputChar == 'j') {vehicle->control->attitudeAndVertPosCtrl(-5,0,angle,height);}
      else if (inputChar == 'l') {vehicle->control->attitudeAndVertPosCtrl(5,0,angle,height);}
      else if (inputChar == 'q') 
      {
        quit = 1;
        bt.off();
      }
      else if (inputChar == 'r')
      {
        vehicle->control->attitudeAndVertPosCtrl(0,0,angle,height);
      }
      else if(inputChar == 'g')
      {
        angle += bearing_angle(globalPosition.latitude, globalPosition.longitude, 38.627089, -90.200203);
	      vehicle->control->attitudeAndVertPosCtrl(0,0,angle,height); // Turn to the target angle
	      distance = 1000 * distanceInKmBetweenEarthCoordinates(globalPosition.latitude, globalPosition.longitude, 38.627089, -90.200203);
	      while (distance >= 5) 
	      {
          vehicle->control->attitudeAndVertPosCtrl(0,-1,angle,height); // Move forward towards the target location with the calculated angle by one meter
          angle += bearing_angle(globalPosition.latitude, globalPosition.longitude, 38.627089, -90.200203); // Recalculate the angle
	        // vehicle->control->attitudeAndVertPosCtrl(0,0,angle,height); // Turn to the target angle
	        distance = 1000 * distanceInKmBetweenEarthCoordinates(globalPosition.latitude, globalPosition.longitude, 38.627089, -90.200203);
          // sleep(1);
	      }
      }
    }
    monitoredLanding(vehicle);
  }

  else if (inputChar == 'b') 
  {
    monitoredTakeoff(vehicle);
    moveByPositionOffset(vehicle, 0, 6, 6, 30);
    moveByPositionOffset(vehicle, 6, 0, -3, -30);
    moveByPositionOffset(vehicle, -6, -6, 0, 0);
    monitoredLanding(vehicle);
  }

    /*! @NOTE: case 'c' only support for m210 V2*/
  else if (inputChar == 'c') 
  {
    /*! Turn off rtk switch */
    ErrorCode::ErrorCodeType ret;
    ret = vehicle->flightController->setRtkEnableSync(
        FlightController::RtkEnabled::RTK_DISABLE, 1);
    if (ret != ErrorCode::SysCommonErr::Success) {
      DSTATUS("Turn off rtk switch failed, ErrorCode is:%8x", ret);
    } else {
      DSTATUS("Turn off rtk switch successfully");
    }
    /*!  Take off */
    monitoredTakeoff(vehicle);
    /*! Move to higher altitude */
    moveByPositionOffset(vehicle, 0, 0, 30, 0);
    /*! Move a short distance*/
    moveByPositionOffset(vehicle, 10, 0, 0, -30);
    /*! Set aircraft current position as new home location */
    setNewHomeLocation(vehicle);
    /*! Set new go home altitude */
    setGoHomeAltitude(vehicle, 50);
    /*! Move to another position */
    moveByPositionOffset(vehicle, 40, 0, 0, 0);
    /*! go home and  confirm landing */
    goHomeAndConfirmLanding(vehicle, 1);
  }

  return 0;
}
