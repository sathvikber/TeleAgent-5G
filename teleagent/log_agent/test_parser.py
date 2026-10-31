from .parser import parse_line


def test_parser():

    test_lines = [

        "09:14:50.900 [CRIT ] SMF msg=SessionModificationFailure error_code=PFCP-0007 action=RELEASE_SESSION",

        "09:31:12.500 [ERROR] SMF msg=PFCPFailure error_code=PFCP-0014",

        "09:31:12.560 [WARN ] AMF msg=UEWarning error_code=NGAP-0021",

        "09:31:12.700 [INFO ] NRF msg=Nnrf_NFDiscovery_Request target=UDM",

        "09:31:12.800 [INFO ] gNB-256 msg=PacketReceived",

        "5G Core Network Event Log",

        "Page 1 of 1"
    ]


    for line in test_lines:

        result = parse_line(line)

        print("\nINPUT:")
        print(line)

        print("\nOUTPUT:")
        print(result)


if __name__ == "__main__":
    test_parser()