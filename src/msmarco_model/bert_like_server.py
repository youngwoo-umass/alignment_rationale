import socketserver
from xmlrpc.server import SimpleXMLRPCRequestHandler
from xmlrpc.server import SimpleXMLRPCServer


class RPCServerWrap:
    def __init__(self, predict_fn):
        self.predict_fn = predict_fn

    def start(self, port):
        class RequestHandler(SimpleXMLRPCRequestHandler):
            rpc_paths = ('/RPC2',)

        class RPCThreading(socketserver.ThreadingMixIn, SimpleXMLRPCServer):
            pass

        print("Preparing server")
        server = RPCThreading(("0.0.0.0", port),
                              requestHandler=RequestHandler,
                              allow_none=True,
                              )
        server.register_introspection_functions()
        server.register_function(self.predict, 'predict')
        print("Waiting")
        server.serve_forever()

    def predict(self, payload):
        return self.predict_fn(payload)



# Example usage :
# proxy = xmlrpc.client.ServerProxy('ingham.cs.umass.edu:8080')
# proxy.predict(payload)
