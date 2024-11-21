import requests
import js2py
from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor

proxies = {
    "http" : "http://127.0.0.1:10201",
    "https" : "http://127.0.0.1:10201"
}

def TL(a):
    k = ""
    b = 406644
    b1 = 3293161072

    jd = "."
    new_b = "+-a^+6"
    Zb = "+-3^+b+-f"

    e, f, g = [], 0, 0

    for g in range(len(a)):
        m = ord(a[g])
        if m < 128:
            e.append(m)
        else:
            if m < 2048:
                e.append((m >> 6) | 192)
            else:
                if 55296 == (m & 64512) and g + 1 < len(a) and 56320 == (ord(a[g + 1]) & 64512):
                    m = 65536 + ((m & 1023) << 10) + (ord(a[g + 1]) & 1023)
                    e.append((m >> 18) | 240)
                    e.append(((m >> 12) & 63) | 128)
                    g += 1
                else:
                    e.append((m >> 12) | 224)
                    e.append(((m >> 6) & 63) | 128)
            e.append((m & 63) | 128)

    a = b
    for f in range(len(e)):
        a += e[f]
        a = RL(a, new_b)
    a = RL(a, Zb)
    a ^= b1 or 0
    if a < 0:
        a = (a & 2147483647) + 2147483648
    a %= 1e6

    return str(int(a)) + jd + str(int(a) ^ int(b))

def RL(a, b):
    t = "a"
    Yb = "+"
    for c in range(0, len(b) - 2, 3):
        d = ord(b[c + 2])
        d = d - 87 if d >= ord(t) else int(chr(d))
        d = js2py.eval_js("{} >>> {}".format(a, d)) if b[c + 1] == Yb else js2py.eval_js("{} << {} ".format(a, d))
        a = (a + d) & 4294967295 if b[c] == Yb else js2py.eval_js("{} ^ {}".format(a, d))
    return a


def _google(url, data):
    param = f"sl={data['langfrom']}&tl={data['langto']}"
    full_url = f"{url}/translate_a/single?client=gtx&{param}&hl=zh-CN&dt=at&dt=bd&dt=ex&dt=ld&dt=md&dt=qca&dt=rw&dt=rm&dt=ss&dt=t&source=bh&ssel=0&tsel=0&kc=1&tk={TL(data['raw'])}&q={data['raw']}"

    response = requests.get(full_url, proxies=proxies)
    if response.status_code != 200:
        raise Exception(f"Request error: {response.status_code}")

    tgt = ""
    for i in range(len(response.json()[0])):
        if response.json()[0][i] and response.json()[0][i][0]:
            tgt += response.json()[0][i][0]
    return tgt

url = "https://translate.google.com"
data = {
    "langfrom": "en",
    "langto": "zh",
    "raw": "TypeError: unsupported operand type(s) for ^: 'float' and 'int'",
}

# _google(url, data)
# print(data["result"])
with open(r"input.txt", "r", encoding="utf-8") as f1, open(r"output_back_trans.txt", "w", encoding="utf-8") as f2:
    lines = f1.readlines()
    for line in tqdm(lines):
        try:
            data = {
                "langfrom": "en",
                "langto": "fr",
                "raw": line,
            }
            trans = _google(url, data)
            data = {
                "langfrom": "fr",
                "langto": "en",
                "raw" : trans
            }
            back_trans = _google(url, data)
            f2.write(back_trans + "\n")
        except:
            f2.write("#### error error #####\n")


