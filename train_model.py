from ultralytics import YOLO

def main():

    model = YOLO("yolo26m.pt")

    model.train(
        data="config.yaml",
        epochs=200, 
        patience=40,
        device=0, # Nvidia 3080ti
        workers=12, # i5 13600kf 14C 20T
        batch=16,
        project=r"C:\Users\lucam\Desktop\Code\blackjack\runs", # Explicitly state dir (saving to wrong dir otherwise) 
        name="card_train"
    )

    model.val(
        conf=0.25,
        project=r"C:\Users\lucam\Desktop\Code\blackjack\runs",
        name="card_val"
    )

if __name__ == "__main__":
    main()