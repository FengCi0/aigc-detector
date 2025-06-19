from app import app
 
if __name__ == "__main__":
    print("启动AIGC检测服务...")
    app.run(host='0.0.0.0', port=5000, debug=True) 