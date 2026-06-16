def benchmark_tokenizer(texts):
    total_bytes = 0
    for item in texts:
        total_bytes += len(item.encode('utf-8'))
    return {'bytes': total_bytes, 'docs': len(texts)}
