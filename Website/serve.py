"""Static file server with HTTP Range request support for video seeking."""

import os
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler

class RangeHTTPRequestHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()

        ctype = self.guess_type(path)
        file_size = os.path.getsize(path)
        range_header = self.headers.get("Range")

        if range_header:
            try:
                ranges = range_header.replace("bytes=", "").split("-")
                start = int(ranges[0]) if ranges[0] else 0
                end = int(ranges[1]) if ranges[1] else file_size - 1
                end = min(end, file_size - 1)
                length = end - start + 1

                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

                f = open(path, "rb")
                f.seek(start)
                f.length = length  # used by copyfile
                return f
            except (ValueError, IndexError):
                pass

        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(file_size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        return open(path, "rb")

    def copyfile(self, source, outputfile):
        length = getattr(source, "length", None)
        if length is not None:
            remaining = length
            buf_size = 64 * 1024
            while remaining > 0:
                chunk = source.read(min(buf_size, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
        else:
            super().copyfile(source, outputfile)

if __name__ == "__main__":
    port = 8080
    server = HTTPServer(("", port), RangeHTTPRequestHandler)
    print(f"Serving on http://localhost:{port} (with Range support)")
    server.serve_forever()
