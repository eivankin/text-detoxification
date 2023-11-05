# Text Detoxification | PMLDL Assignment 1
> Student: Evgenij Ivankin
> 
> E-mail: e.ivankin@innopolis.university
> 
> Group: B21-DS-01

## How to run the solution
### Download and prepare data
1. Download and preprocess filtered ParaNMT-detox corpus:
    ```shell
    python -m src.data.download_paramnt && python -m src.data.preprocess
    ```
2. Download and preprocess profanities list:
   ```shell
   python -m src.data.profanity
   ```