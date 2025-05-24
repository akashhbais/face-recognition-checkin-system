from fastapi import FastAPI, HTTPException
import subprocess

app = FastAPI()

@app.post("/start-recognition/")
def start_recognition():
    try:
        # Run the face recognition script
        process = subprocess.Popen(["python", "face_recognition.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return {"message": "Face recognition started!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting face recognition: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
