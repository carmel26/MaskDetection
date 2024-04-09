import http.server
import socketserver

PORT  = 8080

Handler = http.server.SimpleHTTPRequestHandler
httpd = socketserver.TCPServer(("",PORT), Handler)
print("Server at PORT : ",PORT)
httpd.serve_forever()