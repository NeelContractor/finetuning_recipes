train_on_buffer()
mean_inference_score = inference()

save_model()
if mean_inference_score > best_inference_score:
    print(
        f"New best inference score: {mean_inference_score:.3f}"
    )
    save_model(f"_best_{best_model_id}")
    best_model_id += 1
    best_inference_score = mean_inference_score

