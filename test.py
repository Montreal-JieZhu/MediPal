import subprocess

cmd = [
    "curl",
    "-s",  # silent
    "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "-H", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "-H", "Accept-Language: en-US,en;q=0.9",
    "-H", "Referer: https://medlineplus.gov/",
    "https://medlineplus.gov/druginfo/drug_Ga.html"
]
html = subprocess.check_output(cmd, text=True)
print(html[:5000])  # preview
