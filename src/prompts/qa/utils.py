from loguru import logger


def generated_text2questions(generated_text):
    try:
        lines = generated_text.split("\n")
        lines = [line.strip() for line in lines]
        lines = [line.strip("-") for line in lines]
        lines = [line for line in lines if line and len(line) > 0]
        lines = [line.strip() for line in lines]
        return list(set(lines))
        lines = [line for line in lines if "<question>" in line and "</question>" in line]
        lines = [line[line.find("<question>") + len("<question>") : line.find("</question>")] for line in lines]
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line and len(line) > 0]
        lines = list(set(lines))
    except Exception as e:
        logger.warning(f"Error in processing generated text: {e}")
        lines = []
    return lines
