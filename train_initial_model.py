from ultralytics import YOLO

def main():

    model = YOLO("yolo26m.pt")

    model.train(
        data="config.yaml",
        epochs=150, 
        patience=30,
        device=0, # Nvidia 3080ti
        workers=8, # i5 13600kf 14C 20T
        batch=16,
        project=r"C:\Users\lucam\Desktop\Code\blackjack\runs", # Explicitly state dir (saving to wrong dir otherwise) 
        name="card_train_init"
    )

    model.val(
        data="config.yaml",
        conf=0.25,
        project=r"C:\Users\lucam\Desktop\Code\blackjack\runs",
        name="card_val_init"
    )

if __name__ == "__main__":
    main()