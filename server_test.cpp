#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <stdint.h>
#include <cstdlib>
#include <iostream>

using namespace std;

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

int send_all(int socket, void *buffer, size_t length) {
    char *ptr = (char*) buffer;
    while(length > 0) {
        int num_bytes = send(socket, ptr, length, 0);
        if(num_bytes < 1) 
            return -1;
        ptr += num_bytes;
        length -= num_bytes;
    }
    return 0;
}

int recv_all(int socket, void *buffer, size_t length) {
    char *ptr = (char*) buffer;
    while(length > 0) {
        int num_bytes = recv(socket, ptr, length, 0);
        if(num_bytes < 1) 
            return -1;
        ptr += num_bytes;
        length -= num_bytes;
    }
    return 0;
}

int main(int argc, char *argv[]) {
    const char SOCKET_PATH[] = "/tmp/guardian_socket";
    unlink(SOCKET_PATH);

    Message msg;
    uint16_t signal;
    int sock_fd, conn_fd;
    struct sockaddr_un server_addr, client_addr;
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

    // run visual tracking
    system("python3 vision.py -i test_data/speed_test_person.mp4 -a -s");

    cout << "server: waiting for connection..." << endl;
    if((conn_fd = accept(sock_fd, (struct sockaddr*)&client_addr, (socklen_t*)&client_addr_len)) < 0) {
        cerr << "Socket accept error" << endl;
        exit(1);
    }

    // start visual tracking
    signal = START; 
    if(send_all(conn_fd, &signal, sizeof(signal)) < 0) {
        cerr << "Socket send error" << endl;
        exit(1);
    } 
    
    for(int i = 0; i < 200; ++i) {
        if(recv_all(conn_fd, &msg, sizeof(msg)) < 0) {
            cerr << "Socket recv error" << endl;
            exit(1);
        };
        // cout << msg.type << endl;
        // cout << msg.bbox.xmin << " " << msg.bbox.ymin << " " << msg.bbox.xmax << " " << msg.bbox.ymax << endl;

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

    // pause visual tracking
    signal = STOP; 
    if(send_all(conn_fd, &signal, sizeof(signal)) < 0) {
        cerr << "Socket send error" << endl;
        exit(1);
    }

    sleep(5);
    // start visual tracking
    signal = START; 
    if(send_all(conn_fd, &signal, sizeof(signal)) < 0) {
        cerr << "Socket send error" << endl;
        exit(1);
    }

    sleep(5);
    // terminate visual tracking
    signal = TERMINATE; 
    if(send_all(conn_fd, &signal, sizeof(signal)) < 0) {
        cerr << "Socket send error" << endl;
        exit(1);
    }

    sleep(1);
    close(conn_fd);
    close(sock_fd);

    return 0;
}