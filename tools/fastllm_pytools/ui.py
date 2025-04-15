# Created by bamstone
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import subprocess
import os
import platform
import re

class FastLLMCommandGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("FastLLM命令生成器")
        self.current_file = None
        self.os_type = platform.system()
        self.file_ext = '.cmd' if self.os_type == 'Windows' else '.sh'
        
        self.create_widgets()
        self.setup_file_types()
        self.update_title()

    def setup_file_types(self):
        self.file_types = [
            (f"{self.os_type}脚本文件", f"*{self.file_ext}"),
            ("所有文件", "*.*")
        ]

    def create_widgets(self):
        # 主界面布局
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 工具栏
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill=tk.X, pady=5)
        
        ttk.Button(toolbar, text="打开", command=self.open_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="保存", command=self.save_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="推理", command=self.create_inference_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="下载", command=self.create_download_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="执行", command=self.execute_script).pack(side=tk.LEFT, padx=2)
        
        # 文本编辑区
        self.text_area = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, font=("Consolas", 10))
        self.text_area.pack(fill=tk.BOTH, expand=True)
        
        # 状态栏
        self.status_bar = ttk.Label(self.root, text="就绪", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def update_title(self):
        base_title = "FastLLM命令生成器"
        if self.current_file:
            filename = os.path.basename(self.current_file)
            self.root.title(f"{base_title} - {filename}")
        else:
            self.root.title(base_title)

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=self.file_types)
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    self.text_area.delete('1.0', tk.END)
                    self.text_area.insert(tk.END, content)
                    self.current_file = file_path
                    self.update_title()
                    self.show_status(f"已打开文件: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("错误", f"文件打开失败: {str(e)}")

    def save_file(self):
        if not self.current_file:
            self.current_file = filedialog.asksaveasfilename(
                defaultextension=self.file_ext,
                filetypes=self.file_types
            )
            if not self.current_file:
                return
                
        content = self.text_area.get('1.0', tk.END)
        try:
            with open(self.current_file, 'w') as f:
                f.write(content)
            if self.os_type != 'Windows':
                os.chmod(self.current_file, 0o755)
            self.update_title()
            self.show_status(f"已保存文件: {os.path.basename(self.current_file)}")
            messagebox.showinfo("保存成功", "文件保存成功！")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")

    def execute_script(self):
        if not self.current_file:
            messagebox.showwarning("警告", "请先保存文件再执行")
            return
            
        confirm = messagebox.askyesno("确认", "执行后将退出程序，是否继续？")
        if confirm:
            try:
                if self.os_type == 'Windows':
                    subprocess.Popen(['cmd', '/c', self.current_file], shell=True)
                else:
                    subprocess.Popen(['/bin/bash', self.current_file])
                self.root.destroy()
            except Exception as e:
                messagebox.showerror("执行错误", f"执行失败: {str(e)}")

    def create_inference_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("推理参数设置")
        dialog.geometry("300x300")
        
        # 参数输入框架
        param_frame = ttk.Frame(dialog)
        param_frame.pack(fill=tk.BOTH, padx=10, pady=10)
        
        # 指令类型
        ttk.Label(param_frame, text="*指令类型:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.infer_type = ttk.Combobox(param_frame, values=["run", "webui", "server"], state="readonly")
        self.infer_type.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        self.infer_type.current(0)
        
        # 模型名称
        ttk.Label(param_frame, text="*模型名称:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.model_name = ttk.Entry(param_frame)
        self.model_name.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        
        # 设备参数
        ttk.Label(param_frame, text="device:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.device = ttk.Combobox(param_frame, values=["cpu", "cuda", "numa", "multicuda"])
        self.device.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)
        self.device.bind("<<ComboboxSelected>>", self.toggle_multi_cuda)
        
        # Multicuda附加参数
        self.multi_cuda_frame = ttk.Frame(param_frame)
        ttk.Label(self.multi_cuda_frame, text="Multi-CUDA参数:").pack(side=tk.LEFT)
        self.multi_cuda_args = ttk.Entry(self.multi_cuda_frame)
        self.multi_cuda_args.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.multi_cuda_frame.grid(row=3, columnspan=2, padx=5, pady=5, sticky=tk.EW)
        self.multi_cuda_frame.grid_remove()
        
        # MOE设备
        ttk.Label(param_frame, text="moe_device:").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        self.moe_device = ttk.Combobox(param_frame, values=["cpu", "cuda", "numa"])
        self.moe_device.grid(row=4, column=1, padx=5, pady=5, sticky=tk.EW)
        
        # 线程数
        ttk.Label(param_frame, text="threads:").grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        self.threads = ttk.Entry(param_frame)
        self.threads.grid(row=5, column=1, padx=5, pady=5, sticky=tk.EW)
        
        # 数据类型
        ttk.Label(param_frame, text="dtype:").grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)
        self.dtype = ttk.Combobox(param_frame, values=["int4", "int8"])
        self.dtype.grid(row=6, column=1, padx=5, pady=5, sticky=tk.EW)
        
        # 生成按钮
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="生成命令", 
                 command=lambda: self.generate_inference_command(dialog)).pack(padx=5)

    def toggle_multi_cuda(self, event):
        if self.device.get() == "multicuda":
            self.multi_cuda_frame.grid()
        else:
            self.multi_cuda_frame.grid_remove()

    def generate_inference_command(self, dialog):
        # 参数验证
        model = self.model_name.get().strip()
        if not model:
            messagebox.showerror("错误", "必须输入模型名称")
            return
        
        # 验证线程数
        threads = self.threads.get().strip()
        if threads and not re.match(r'^\d+$', threads):
            messagebox.showerror("错误", "线程数必须是整数")
            return
        
        # 验证multicuda参数
        device = self.device.get()
        if device == "multicuda" and not self.multi_cuda_args.get().strip():
            messagebox.showerror("错误", "使用multicuda时必须提供附加参数")
            return

        # 构建参数列表
        params = []
        if device:
            param = f"--device {device}"
            if device == "multicuda":
                param += f":{self.multi_cuda_args.get().strip()}"
            params.append(param)
        if self.moe_device.get():
            params.append(f"--moe_device {self.moe_device.get()}")
        if threads:
            params.append(f"--threads {threads}")
        if self.dtype.get():
            params.append(f"--dtype {self.dtype.get()}")
        
        # 组合完整命令
        command = f"ftllm {self.infer_type.get()} {model} {' '.join(params)}"
        self.text_area.insert(tk.END, command + "\n")
        dialog.destroy()
        self.show_status("推理命令已生成")

    def create_download_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("下载参数设置")
        dialog.geometry("300x150")
        
        # 下载参数框架
        param_frame = ttk.Frame(dialog)
        param_frame.pack(fill=tk.BOTH, padx=10, pady=10)
        
        # 模型名称
        ttk.Label(param_frame, text="*模型名称:").grid(row=0, column=0, padx=5, pady=10, sticky=tk.W)
        self.download_model = ttk.Entry(param_frame)
        self.download_model.grid(row=0, column=1, padx=5, pady=10, sticky=tk.EW)
        
        # 生成按钮
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="生成命令", 
                 command=lambda: self.generate_download_command(dialog)).pack(padx=5)

    def generate_download_command(self, dialog):
        model = self.download_model.get().strip()
        if not model:
            messagebox.showerror("错误", "必须输入模型名称")
            return
        
        command = f"ftllm download {model}"
        self.text_area.insert(tk.END, command + "\n")
        dialog.destroy()
        self.show_status("下载命令已生成")

    def show_status(self, message):
        self.status_bar.config(text=message)
        self.root.after(5000, lambda: self.status_bar.config(text="就绪"))

def FastllmStartUI():
    root = tk.Tk()
    root.geometry("800x600")
    app = FastLLMCommandGenerator(root)
    root.mainloop()

if __name__ == "__main__":
    FastllmStartUI()