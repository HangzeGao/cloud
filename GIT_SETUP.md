# Git 版本管理设置指南

## 1. 初始化 Git 仓库

在项目根目录执行：

```bash
cd /Users/hangzegao/PycharmProjects/MyCloudSense

# 初始化 git 仓库
git init

# 配置用户信息
git config user.email "your-email@example.com"
git config user.name "Your Name"
```

## 2. 创建 .gitignore 文件

```bash
# 创建 .gitignore
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

# Data
data/
Data/

# Logs
*.log
tensorboard/
.cursor/

# OS
.DS_Store
Thumbs.db
EOF
```

## 3. 首次提交代码

```bash
# 添加所有文件到暂存区
git add .

# 查看状态
git status

# 首次提交
git commit -m "Initial commit: CloudSense-Net cloud segmentation model"
```

## 4. 创建开发分支

```bash
# 创建并切换到开发分支
git checkout -b dev

# 在 dev 分支上进行开发
# ... 修改代码 ...

# 提交修改
git add .
git commit -m "feat: add MPS support for Apple Silicon"
```

## 5. 常用 Git 命令

### 查看状态
```bash
git status                    # 查看工作区状态
git log --oneline            # 查看提交历史（简洁）
git log --graph --oneline    # 查看分支图
```

### 提交更改
```bash
git add <file>               # 添加特定文件
git add .                    # 添加所有更改
git commit -m "message"      # 提交更改
```

### 分支操作
```bash
git branch                   # 查看本地分支
git branch -a                # 查看所有分支
git checkout <branch>        # 切换分支
git checkout -b <new-branch> # 创建并切换分支
git merge <branch>           # 合并分支到当前分支
```

### 撤销操作
```bash
git checkout -- <file>       # 撤销文件修改
git reset HEAD <file>        # 取消暂存
git reset --soft HEAD~1      # 撤销上次提交（保留更改）
git reset --hard HEAD~1      # 撤销上次提交（丢弃更改）
```

### 查看差异
```bash
git diff                     # 查看未暂存的更改
git diff --cached            # 查看已暂存的更改
git diff HEAD~1              # 查看与上次提交的差异
```

## 6. 推荐工作流程

### 功能开发流程
```bash
# 1. 从主分支创建功能分支
git checkout main
git pull
git checkout -b feature/new-feature

# 2. 开发并提交
git add .
git commit -m "feat: implement new feature"

# 3. 合并回主分支
git checkout main
git merge feature/new-feature
```

### Bug 修复流程
```bash
# 1. 创建修复分支
git checkout -b fix/bug-name

# 2. 修复并提交
git add .
git commit -m "fix: resolve dimension mismatch in decoder"

# 3. 合并
git checkout main
git merge fix/bug-name
```

## 7. 保存当前修复的提交示例

```bash
# 添加修复后的文件
git add models/decoders/segformer_decoder.py
git add models/cloudseg_model.py

# 提交
git commit -m "fix: remove hardcoded scale_factor=4 in decoder

- Decoder output was 2x input size instead of matching input
- c1 feature is 1/2 of original, not 1/4 as assumed
- Moved final upsampling to CloudSenseNet.forward() for consistency
- Fixes RuntimeError: tensor size mismatch in loss calculation"
```

## 8. 连接到远程仓库（可选）

```bash
# 添加远程仓库
git remote add origin https://github.com/yourusername/MyCloudSense.git

# 推送到远程
git push -u origin main

# 后续推送
git push
```

## 当前状态总结

已完成：
- ✅ 修复了 decoder 尺寸不匹配问题
- ✅ 添加了 MPS 支持
- ✅ 创建了可配置架构

建议下一步：
1. 初始化 git 仓库
2. 创建 .gitignore
3. 提交初始代码
4. 创建一个提交记录当前的修复
