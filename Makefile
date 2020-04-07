.PHONY: data

data: data/test_data.avi data/test_index.yaml data/test_scores.h5 data/test_model.p data/mock_model.p

data/test_data.avi:
	aws s3 cp s3://moseq2-testdata/viz/ data/ --request-payer=requester --recursive

data/test_index.yaml:
	aws s3 cp s3://moseq2-testdata/pca data/ --request-payer=requester --recursive

data/test_scores.h5:
	aws s3 cp s3://moseq2-testdata/viz/ data/_pca/ --request-payer=requester --recursive

data/test_model.p:
	aws s3 cp s3://moseq2-testdata/model data/ --request-payer=requester --recursive

data/mock_model.p:
	aws s3 cp s3://moseq2-testdata/viz data/ --request-payer=requester --recursive