# GitHub 上传指南

本项目的实际代码目录是：

```powershell
C:\Users\ASUS\Desktop\项目1\UavNetSim-master (3)\extracted\UavNetSim-master
```

建议在这个目录中初始化 Git 仓库并上传到 GitHub。不要从上一层目录上传，否则会把外层压缩包和中间目录结构也带上。

## 1. 安装 Git

如果 `git --version` 提示找不到命令，请先安装 Git for Windows：

<https://git-scm.com/download/win>

安装完成后，重新打开 PowerShell。

## 2. 初始化并提交

```powershell
cd "C:\Users\ASUS\Desktop\项目1\UavNetSim-master (3)\extracted\UavNetSim-master"
git init
git add .
git commit -m "Initial commit"
```

## 3. 创建 GitHub 仓库

在 GitHub 新建一个空仓库，不要勾选自动生成 README、.gitignore 或 LICENSE，因为项目中已经有这些文件。

假设你的仓库地址是：

```text
https://github.com/你的用户名/你的仓库名.git
```

## 4. 关联远程仓库并推送

```powershell
git branch -M main
git remote add origin https://github.com/你的用户名/你的仓库名.git
git push -u origin main
```

## 5. 上传前检查

运行下面命令确认将要提交的文件：

```powershell
git status
git ls-files
```

当前 `.gitignore` 已排除 Python 缓存、IDE 配置、运行日志、压缩包和生成的 GIF 动画，避免把很大的临时文件上传到 GitHub。
