def main() -> int:
    try:
        import torch
    except Exception:
        print("torch is not importable")
        return 1

    n = 0
    try:
        n = int(torch.cuda.device_count())
    except Exception:
        n = 0

    print(n)
    if n > 0:
        try:
            for i in range(n):
                print(i, torch.cuda.get_device_name(i))
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
