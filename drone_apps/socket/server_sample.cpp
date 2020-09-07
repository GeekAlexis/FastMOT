#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <unistd.h>
#include <stdint.h>
#include <cstdlib>
#include <iostream>

using namespace std;

/* 
constants for different message types
*/
#define BBOX 0
#define TARGET_NOT_FOUND 1
#define TARGET_LOST 2
#define START 3
#define STOP 4
#define TERMINATE 5

struct BoundingBox {
    int16_t xmin;
    int16_t ymin;
    int16_t xmax;
    int16_t ymax;
};

struct Message {
    uint16_t type;
    BoundingBox bbox;
};

int send_signal(int socket, uint16_t *signal) {
    *signal = htons(*signal);
    size_t length = sizeof(uint16_t);
    char *ptr = (char*) signal;
    while(length > 0) {
        int num_bytes = send(socket, ptr, length, 0);
        if(num_bytes < 0) 
            return -1;
        ptr += num_bytes;
        length -= num_bytes;
    }
    return 0;
}

int recv_msg(int socket, Message *msg) {
    size_t length = sizeof(Message);
    char *ptr = (char*) msg;
    while(length > 0) {
        int num_bytes = recv(socket, ptr, length, 0);
        if(num_bytes < 0) 
            return -1;
        ptr += num_bytes;
        length -= num_bytes;
    }
    msg->type = ntohs(msg->type);
    msg->bbox.xmin = ntohs(msg->bbox.xmin);
    msg->bbox.ymin = ntohs(msg->bbox.ymin);
    msg->bbox.xmax = ntohs(msg->bbox.xmax);
    msg->bbox.ymax = ntohs(msg->bbox.ymax);
    return 0;
}

int main(int argc, char *argv[]) {
    const char SOCKET_PATH[] = "/tmp/fastmot_socket";
    unlink(SOCKET_PATH);

    Message msg;
    uint16_t signal;
    int sock_fd, conn_fd;
    sockaddr_un server_addr, client_addr;
    int client_addr_len = sizeof(client_addr);
    server_addr.sun_family = AF_UNIX;
    strncpy(server_addr.sun_path, SOCKET_PATH, sizeof(server_addr.sun_path));

    if((sock_fd = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
        cerr << "Socket creation error" << endl;
        exit(1);
    }
    if(bind(sock_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        cerr << "Socket bind error" << endl;
        exit(1);
    }
    if(listen(sock_fd, 1) < 0) {
        cerr << "Socket listen error" << endl;
        exit(1);
    }

    // Run Python visual tracking in the background
    // change the file name after "-i" to a diffent video if needed
    system("python3 vision.py -i test_data/speed_test_person.mp4 -a -s &");

    cout << "server: waiting for connection..." << endl;
    if((conn_fd = accept(sock_fd, (struct sockaddr*)&client_addr, (socklen_t*)&client_addr_len)) < 0) {
        cerr << "Socket accept error" << endl;
        exit(1);
    }
    cout << "server: connected" << endl;

    /*
    Examples for sending and receiving messages are shown below, feel free to change these
    */

    // start visual tracking
    signal = START; 
    if(send_signal(conn_fd, &signal) < 0) {
        cerr << "Socket send error" << endl;
        exit(1);
    } 
    
    // example receive loop
    for(int i = 0; i < 300; ++i) {
        // receive 300 times for example
        if(recv_msg(conn_fd, &msg) < 0) {
            cerr << "Socket recv error" << endl;
            exit(1);
        };

        if(msg.type == BBOX) {
            cout << "server: " << msg.bbox.xmin << " " << msg.bbox.ymin << " " << msg.bbox.xmax << " " << msg.bbox.ymax << endl;
            // if target is acquired, this will be sent every frame 
            // TODO: process bbox coordinates and generate control signals to track target.
        }   
        else if(msg.type == TARGET_NOT_FOUND) {
            cout << "server: target not found" << endl;
            // sent if no target is found for 100 frames
            // TODO: turn a little bit to search for a new target.
        }
        else if(msg.type == TARGET_LOST) {
            cout << "server: target lost" << endl;
            // only sent immediately after losing track
            // TODO: stop following, turn a little to search for a new target. If still no target found, return to GPS waypoint.
        }
    }

    // pause visual tracking when not used to save power
    signal = STOP; 
    if(send_signal(conn_fd, &signal) < 0) {
        cerr << "Socket send error" << endl;
        exit(1);
    }

    // start visual tracking again
    signal = START; 
    if(send_signal(conn_fd, &signal) < 0) {
        cerr << "Socket send error" << endl;
        exit(1);
    }

    sleep(5);
    // pause visual tracking after 5 seconds
    signal = STOP; 
    if(send_signal(conn_fd, &signal) < 0) {
        cerr << "Socket send error" << endl;
        exit(1);
    }

    // terminate visual tracking program at the end
    signal = TERMINATE; 
    if(send_signal(conn_fd, &signal) < 0) {
        cerr << "Socket send error" << endl;
        exit(1);
    }

    sleep(1);
    close(conn_fd);
    close(sock_fd);

    return 0;
}