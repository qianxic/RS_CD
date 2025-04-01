import io
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import os
from typing import List

from app.models.model import ChangeDetectionModel
from app.utils import process_images, decode_results, encode_image_to_base64

# 获取环境变量中的模型路径，默认为app/models/model.pth
MODEL_PATH = os.environ.get("MODEL_PATH", "app/models/model.pth")

app = FastAPI(title="变化检测模型API", description="用于处理遥感图像变化检测的API")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源，生产环境中应该限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加载模型
model = None

@app.on_event("startup")
async def startup_event():
    global model
    model = ChangeDetectionModel()
    try:
        model.load_model(MODEL_PATH)
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        # 我们不会立即抛出异常，而是让API可以启动，但健康检查会反映这个问题

@app.get("/")
def read_root():
    return {"message": "变化检测模型API已成功运行", "model_status": "已加载" if model else "未加载"}

@app.post("/detect_change/")
async def detect_change(
    image_before: UploadFile = File(...),
    image_after: UploadFile = File(...),
):
    """
    接收两张图像（前后时相），返回变化检测结果
    
    - **image_before**: 前时相图像文件
    - **image_after**: 后时相图像文件
    
    返回:
    - 变化检测结果，包括success状态和变化图（base64编码的图像）
    """
    if not model:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    try:
        # 读取上传的图像
        before_image = Image.open(io.BytesIO(await image_before.read()))
        after_image = Image.open(io.BytesIO(await image_after.read()))
        
        # 处理图像
        processed_data = process_images(before_image, after_image)
        
        # 进行预测
        with torch.no_grad():
            results = model.predict(processed_data)
        
        # 解码结果
        change_map = decode_results(results)
        
        # 将结果编码为base64字符串
        change_map_base64 = encode_image_to_base64(change_map)
        
        # 返回结果
        return JSONResponse(content={
            "success": True,
            "change_map_base64": change_map_base64,
            "message": "变化检测完成"
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理过程中出错: {str(e)}")

@app.get("/health")
def health_check():
    """健康检查接口"""
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    } 