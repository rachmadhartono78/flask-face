from app import create_app
import logging

app = create_app()

if __name__ == '__main__':
    from waitress import serve
    logging.basicConfig(level=logging.INFO)
    serve(app, host='0.0.0.0', port=5000)
