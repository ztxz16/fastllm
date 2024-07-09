try:
    import streamlit as st
except:
    print("Plase install streamlit-chat. (pip install streamlit-chat)")
    exit(0)

import os
import sys

if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    web_demo_path = os.path.join(current_path, 'web_demo.py')
    port = ""
    for i in range(len(sys.argv)):
        if sys.argv[i] == "--port":
            port = "--server.port " + sys.argv[i + 1]
        if sys.argv[i] == "--help" or sys.argv[i] == "-h":
            os.system("python3 " + web_demo_path + " --help")
            exit(0)
    os.system("streamlit run " + port + " " + web_demo_path + ' -- ' + ' '.join(sys.argv[1:]))