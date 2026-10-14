from graph_rag_engine import HyrbidGraphVectorRAG

sample_tech_doc="""
The PaymentService module relies directly on AuthGateway for user JWT validation.

AuthGateway utilizes RedisCache to store session tokens and prevent rate limiting.

When RedisCache experiences high latency, the RateLimiter service automatically triggers failover to PostgreSQL.

PostgreSQL sends critical alert metrics to the Datadog Monitoring System.

"""

def main():
    rag = HyrbidGraphVectorRAG()
    print("Ingesting Document & Constructing Knowledge Graph")
    rag.add_document(sample_tech_doc)

    query= "What happens downstream when PaymentService authentication fails due to RedicCache latency?"

    answer=rag.hybird_query(query)

    print("\n====FINAL HYBRID RAG ANSWER====")
    print(answer)

if __name__=="__main__":
    main()