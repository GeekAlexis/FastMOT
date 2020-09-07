import time
import os
import socket
import struct
import errno


class MsgType:
    """
    enumeration type and function for socket messages
    """
    BBOX, TARGET_NOT_FOUND, TARGET_LOST, START, STOP, TERMINATE = (i for i in range(6))


# def convert_bbox_to_bytes(bbox):
    # length = MSG_LENGTH // 4
    # return b''.join(int(coord).to_bytes(length, byteorder='big', signed=True) for coord in bbox.tf_rect())


def serialize_to_msg(msg_type, bbox=None):
    if bbox is None:
        return struct.pack('!H8x', msg_type)
    return struct.pack('!Hhhhh', msg_type, *bbox.tf_rect())


def parse_from_msg(msg):
    return struct.unpack('!H', msg)[0]


MSG_LENGTH = 2
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.connect('/tmp/fastmot_socket')
sock.setblocking(False)
buffer = b''


for i in range(300):
    try:
        buffer += sock.recv(MSG_LENGTH - len(buffer))
        # msg = sock.recv(MSG_LENGTH)
    except OSError as err:
        if err.args[0] != errno.EAGAIN and err.args[0] != errno.EWOULDBLOCK:
            raise
    else:
        if len(buffer) == MSG_LENGTH:
            signal = parse_from_msg(buffer)
            print(signal)
            buffer = b''
            # print('client', msg_type)
            if signal == MsgType.START:
                print('client: start')
            elif signal == MsgType.STOP:
                print('client: stop')
            elif signal == MsgType.TERMINATE:
                print('client: terminate')
                break
            

    # if analytics.status == Analytics.Status.TARGET_ACQUIRED:
    #     msg = serialize_to_msg(MsgType.BBOX, analytics.get_target_bbox())
    #     sock.sendall(msg)
    if i == 50:
        msg = serialize_to_msg(MsgType.TARGET_NOT_FOUND)
        sock.sendall(msg)
    elif i == 80:
        msg = serialize_to_msg(MsgType.TARGET_LOST)
        sock.sendall(msg)
    
    time.sleep(0.1)

sock.close()