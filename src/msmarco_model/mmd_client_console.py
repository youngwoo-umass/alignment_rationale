from msmarco_model.client_lib import BERTClient
from cpath import MMD_PORT


def main():
    client = BERTClient("http://localhost", MMD_PORT, 512)
    while True:
        sent1 = input("Query: ")
        sent2 = input("Document: ")
        ret = client.request_single(sent1, sent2)
        print((sent1, sent2))
        print(ret)


if __name__ == "__main__":
    main()
