# Script to create the database
import json

# Sample data based on your example
qa_data = [
    {
        "question": "Install and run Visual Studio Code. In your Terminal (or Command Prompt), type code -s and press Enter. Copy and paste the entire output below.",
        "answer": "Version: Code 1.96.4 (cd4ee3b1c348a13bafd8f9ad8060705f6d4b9cba, 2025-01-16T00:16:19.038Z)\nOS Version: Windows_NT x64 10.0.26100\nCPUs: 13th Gen Intel(R) Core(TM) i5-13500HX (20 x 2688)\nMemory (System): 15.73GB (5.52GB free)\nVM: 0%\nScreen Reader: no\nProcess Argv: --crash-reporter-id 478d37a0-e63d-4361-a319-53bd3b6dc91d\nGPU Status: 2d_canvas: enabled\ncanvas_oop_rasterization: enabled_on\ndirect_rendering_display_compositor: disabled_off_ok\ngpu_compositing: enabled\nmultiple_raster_threads: enabled_on\nopengl: enabled_on\nrasterization: enabled\nraw_draw: disabled_off_ok\nskia_graphite: disabled_off\nvideo_decode: enabled\nvideo_encode: enabled\nvulkan: disabled_off\nwebgl: enabled\nwebgl2: enabled\nwebgpu: enabled\nwebnn: disabled_off\n\nCPU % Mem MB PID Process\n0 148 1032 code main\n2 461 5088 extensionHost [1]\n0 111 7932 electron-nodejs (server.js )\n0 7 16888 c:\\Users\\asp61\\.vscode\\extensions\\ms-python.python-2024.22.2-win32-x64\\python-env-tools\\bin\\pet.exe server\n0 7 15432 C:\\WINDOWS\\system32\\conhost.exe 0x4\n0 19 28948 \"c:\\Users\\asp61\\AppData\\Local\\Programs\\Microsoft VS Code\\resources\\app\\node_modules\\@vscode\\ripgrep\\bin\\rg.exe\" --files --hidden --case-sensitive --no-require-git -g !**/.git -g !**/.svn -g !**/.hg -g !**/CVS -g !**/.DS_Store -g !**/Thumbs.db -g !**/node_modules -g !**/bower_components -g !**/*.code-search --no-ignore-parent --follow --no-config --no-ignore-global\n0 7 14456 C:\\WINDOWS\\system32\\conhost.exe 0x4\n1 153 8568 gpu-process\n0 110 11292 fileWatcher [1]\n0 52 15804 utility-network-service\n0 34 18956 crashpad-handler\n1 390 20356 window [1] (grrrr.ipynb - ML android - Visual Studio Code)\n0 140 24476 window\n0 178 27840 shared-process\n0 119 28824 ptyHost\n0 8 5772 conpty-agent\n0 83 20504 C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe -noexit -command \"try { . \\\"c:\\Users\\asp61\\AppData\\Local\\Programs\\Microsoft VS Code\\resources\\app\\out\\vs\\workbench\\contrib\\terminal\\common\\scripts\\shellIntegration.ps1\\\" } catch {}\"\n0 84 27392 C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe -noexit -command \"try { . \\\"c:\\Users\\asp61\\AppData\\Local\\Pr 0 106 29252 electron-nodejs (cli.js )\n0 134 15132 \"C:\\Users\\asp61\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe\" -s\n1 97 14700 gpu-process\n0 87 20272 crashpad-handler\n0 93 21644 utility-network-service\n0 8 29004 conpty-agent\n\nWorkspace Stats:\n| Window (grrrr.ipynb - ML android - Visual Studio Code)\n| Folder (ML android): more than 23595 files\n| File types: json(8090) jpg(6191) JPG(4405) xml(665) txt(2) zip(2) md(1)\n| pdf(1) csv(1) yml(1)\n| Conf files"
    },
    {
        "question": "Download and unzip file abcd.zip which has a single extract.csv file inside. What is the value in the \"answer\" column of the CSV file?",
        "answer": "1234567890"
    }
    # Add more question-answer pairs here
]

with open('qa_database.json', 'w') as f:
    json.dump(qa_data, f, indent=2)
