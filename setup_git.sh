#!/bin/bash
# CloudSense-Net Git 初始化脚本

echo "🚀 Setting up Git for CloudSense-Net..."

# 检查是否在正确目录
if [ ! -f "train.py" ]; then
    echo "❌ Error: Please run this script from /Users/hangzegao/PycharmProjects/MyCloudSense"
    exit 1
fi

# 初始化 git 仓库
echo "📦 Initializing git repository..."
git init

# 配置用户信息
echo "👤 Configuring git user..."
git config user.email "dev@cloudsense.net"
git config user.name "CloudSense Developer"

# 创建 .gitignore
echo "📝 Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# Virtual environments
.venv/
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Model checkpoints and logs
experiments/
experiments_*/
checkpoints/
*.pth
*.ckpt

# Data (large files)
data/
Data/
*.jpg
*.png
*.tif

# Logs
*.log
tensorboard/
.cursor/

# OS
.DS_Store
Thumbs.db
EOF

# 添加文件到暂存区
echo "➕ Adding files to staging area..."
git add .

# 查看状态
echo "📊 Git status:"
git status

# 首次提交
echo "💾 Creating initial commit..."
git commit -m "Initial commit: CloudSense-Net cloud segmentation model

Features:
- Configurable architecture (Swin/ConvNeXt/EfficientNet/ResNet)
- Semantic enhancement module (SkySense++ MSL)
- Multiple fusion options (FPN/BiFPN/ASPP)
- Various decoders (UNet++/DeepLabV3+/SegFormer/UperNet)
- Apple MPS support for Apple Silicon
- Flexible loss functions (Dice/BCE/Focal/Boundary/Tversky)
- Arbitrary size inference (sliding window/multi-scale/whole image)
"

# 创建开发分支
echo "🌿 Creating dev branch..."
git checkout -b dev

# 标记当前修复
git add models/decoders/segformer_decoder.py models/cloudseg_model.py
git commit -m "fix: resolve dimension mismatch in SegFormerDecoder

- Removed hardcoded scale_factor=4 upsampling in decoder
- c1 feature was 1/2 of original, not 1/4 as assumed
- Moved final upsampling to CloudSenseNet.forward()
- Fixes RuntimeError: tensor size mismatch in loss calculation"

echo ""
echo "✅ Git setup complete!"
echo ""
echo "Current branches:"
git branch -a
echo ""
echo "Recent commits:"
git log --oneline -5
echo ""
echo "📚 Next steps:"
echo "  - Work on 'dev' branch: git checkout dev"
echo "  - Create feature branches: git checkout -b feature/your-feature"
echo "  - View status: git status"
echo "  - View log: git log --oneline"
