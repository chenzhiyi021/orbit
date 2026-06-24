"""
Convert Hugging Face models to Megatron checkpoint format
使用 spawn 启动方式避免 multiprocessing 问题
"""

import os
import multiprocessing as mp

# 关键：必须在 __main__ 代码执行前就设置启动方式
# 这行代码必须在 if __name__ == '__main__' 之前
mp.set_start_method('spawn', force=True)

import torch
from megatron.bridge import AutoBridge

def main():
    # 设置环境变量（可选，用于调试）
    os.environ['NVTE_ASYNC_SAVE'] = '0'  # 禁用异步保存
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 避免 tokenizer 警告
    
    # 输入输出路径
    HF_MODEL_PATH = "/mnt/L202500431/models/qwen3-4b-instruct-2507"
    MEGATRON_SAVE_PATH = "/mnt/L202500431/models/megatron_ckpt/qwen3-4b-instruct-2507"
    
    # 确保输出目录存在
    os.makedirs(MEGATRON_SAVE_PATH, exist_ok=True)
    
    print(f"🚀 开始转换...")
    print(f"   源模型: {HF_MODEL_PATH}")
    print(f"   目标路径: {MEGATRON_SAVE_PATH}")
    
    # 执行转换
    try:
        AutoBridge.import_ckpt(
            hf_model_id=HF_MODEL_PATH,
            megatron_path=MEGATRON_SAVE_PATH,
        )
        print("✅ 转换完成！")
        
        # 验证转换结果
        print("\n📂 生成的检查点文件:")
        for root, dirs, files in os.walk(MEGATRON_SAVE_PATH):
            level = root.replace(MEGATRON_SAVE_PATH, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            sub_indent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # 只显示前5个文件
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)
                print(f"{sub_indent}{file} ({file_size:.2f} MB)")
            if len(files) > 5:
                print(f"{sub_indent}... 共 {len(files)} 个文件")
                
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        raise

if __name__ == '__main__':
    main()