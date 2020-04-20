import socket
from pathlib import Path
import time


MSG_LENGTH = 16


class Msg:
    """
    enumeration type and function for socket messages
    """
    TARGET_NOT_FOUND, TARGET_LOST, START, STOP, TERMINATE = ((i).to_bytes(MSG_LENGTH, byteorder='big') for i in range(5))


def convert_bytes_to_bbox(byte_arr):
    length = MSG_LENGTH // 4
    return [int.from_bytes(byte_arr[i:i + length], byteorder='big', signed=True) for i in range(0, MSG_LENGTH, length)]


# delete socket path if it exists
address = Path('/tmp/guardian_socket')
if address.exists():
    address.unlink()

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

sock.bind(str(address))
sock.listen(1)
print('server: Waiting for connection...')
conn, client_addr = sock.accept()
# conn.setblocking(False)
print('server: connection from ', client_addr)

conn.sendall(Msg.START)
for i in range(200):
    msg = conn.recv(MSG_LENGTH)
    if msg:
        #print('server', msg)
        if msg == Msg.TARGET_NOT_FOUND:
            print('server: target not found')
        elif msg == Msg.TARGET_LOST:
            print('server: target lost')
        else:
            print('server:', convert_bytes_to_bbox(msg))

conn.sendall(Msg.STOP)
time.sleep(5)
conn.sendall(Msg.START)
time.sleep(5)
conn.sendall(Msg.TERMINATE)
time.sleep(1)
conn.close()
