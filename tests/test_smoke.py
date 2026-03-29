from loma import LoMa

def test_smoke():
    model = LoMa(LoMa.Cfg(compile=False))

if __name__ == "__main__":
    test_smoke()