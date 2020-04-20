#include <sys/socket.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>

using namespace std;

#define TARGET_NOT_FOUND 0
#define TARGET_LOST 1
#define START 2
#define STOP 3
#define TERMINATE 4

struct BoundingBox {
    int32_t xmin;
    int32_t ymin;
    int32_t xmax;
    int32_t ymax;
};

int main(int argc, char *argv[]) {
    const char SOCKET_PATH[] = "/tmp/guardian_socket";
    unlink(SOCKET_PATH);

    


    return 0;
}