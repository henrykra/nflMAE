import train


if __name__ == "__main__": 
    config = train.parse_config()

    config['force'] = False # ensure same training sets

    for model_name in ['my', 'stt']: 
        config['model']['model'] = model_name

        train.main(config)