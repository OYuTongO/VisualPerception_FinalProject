from pythonosc import udp_client


def main():
    client = udp_client.SimpleUDPClient("127.0.0.1", 8000)
    client.send_message("/asl/trigger", 1)


if __name__ == "__main__":
    main()
