if __name__ == "__main__":
    try:
        from court import LLMCourt

        court = LLMCourt()
    except ModuleNotFoundError as e:
        print(f"Missing dependency: {e.name}")
        print(
            "\nSetup example:\n"
            "  python3 -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  python3 -m pip install -r requirements.txt\n"
        )
        raise SystemExit(1)
    except FileNotFoundError as e:
        print(e)
        print(
            "\nSetup example:\n"
            "  python3 -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  python3 -m pip install -r requirements.txt\n"
            "  python3 preprocess.py --cases data/hanrei_data --laws data/hourei_data\n"
            "  python3 emb_db/build_index.py\n"
        )
        raise SystemExit(1)

    case = "AはBに100ドルを貸したが返済されていない"

    result = court.run(case)

    print("\n===== FINAL =====\n")
    print(result["judgment"])
