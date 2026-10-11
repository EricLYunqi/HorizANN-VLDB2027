#include <gtest/gtest.h>
#include <grpcpp/grpcpp.h>
#include <thread>
#include "include/vec_bytes.h"
#include "src/master/master_server.h"
#include "mock/MockQueryRouter.h"
#include "mock/MockAggregator.h"
#include "src/worker/worker_server.h"
#include "src/worker/cluster_index.h"

namespace ann {

static std::pair<std::unique_ptr<grpc::Server>, std::string> startServer(grpc::Service* service) {
    int selected_port = 0;
    grpc::ServerBuilder builder;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &selected_port);
    builder.RegisterService(service);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    std::string address = std::string("127.0.0.1:") + std::to_string(selected_port);
    return {std::move(server), address};
}

class MasterWorkerIntegrationTest : public ::testing::Test, public ::testing::WithParamInterface<int> {
protected:
    void SetUp() override {
        // Number of workers provided by parameterization
        int num_workers = GetParam();

        worker_services_.reserve(static_cast<size_t>(num_workers));
        worker_servers_.reserve(static_cast<size_t>(num_workers));
        worker_addrs_.reserve(static_cast<size_t>(num_workers));

        for (int i = 0; i < num_workers; ++i) {
            auto cluster_index = std::make_unique<ClusterIndex<float>>(/*dim=*/4, /*max_elements=*/1000, /*M=*/16, /*ef_construction=*/200);
            auto svc = std::make_unique<WorkerServiceImpl<float>>(std::move(cluster_index), /*worker_id=*/i + 1, /*dim=*/4, /*max_elements=*/1000);
            auto [server, addr] = startServer(svc.get());
            worker_addrs_.push_back(addr);
            worker_servers_.push_back(std::move(server));
            worker_services_.push_back(std::move(svc));
        }

        // Create master with all workers
        auto router = std::make_unique<MockQueryRouter>(/*dim=*/4, worker_addrs_);
        auto aggregator = std::make_unique<MockAggregator>();
        master_svc_ = std::make_unique<MasterServiceImpl>(std::move(router), std::move(aggregator), std::deque<std::string>{});
        
        auto [master_server, master_addr] = startServer(master_svc_.get());
        master_server_ = std::move(master_server);
        master_addr_ = master_addr;

        // Create client stub to master
        auto channel = grpc::CreateChannel(master_addr_, grpc::InsecureChannelCredentials());
        client_stub_ = MasterService::NewStub(channel);
    }

    void TearDown() override {
        if (master_server_) master_server_->Shutdown();
        for (auto& s : worker_servers_) if (s) s->Shutdown();
    }

    std::unique_ptr<MasterServiceImpl> master_svc_;
    std::unique_ptr<grpc::Server> master_server_;
    std::unique_ptr<MasterService::Stub> client_stub_;
    std::vector<std::unique_ptr<WorkerServiceImpl<float>>> worker_services_;
    std::vector<std::unique_ptr<grpc::Server>> worker_servers_;
    std::vector<std::string> worker_addrs_;
    std::string master_addr_;
};

TEST_P(MasterWorkerIntegrationTest, InsertSuccess) {
    InsertData ins;
    ins.set_id(99);
    { static const float v[] = {1.0f, 2.0f, 3.0f, 4.0f}; set_float_bytes(ins.mutable_vector_data(), v, 4); }

    InsertAck ack;
    grpc::ClientContext ctx;
    auto status = client_stub_->Insert(&ctx, ins, &ack);
    
    EXPECT_TRUE(status.ok()) << "Insert failed: " << status.error_message();
    EXPECT_TRUE(ack.success()) << "Insert should succeed with valid vector";
}

TEST_P(MasterWorkerIntegrationTest, InsertEmptyVectorFails) {
    InsertData ins;
    ins.set_id(100);
    // No vectors added - empty vector

    InsertAck ack;
    grpc::ClientContext ctx;
    auto status = client_stub_->Insert(&ctx, ins, &ack);
    
    EXPECT_TRUE(status.ok()) << "Insert RPC should succeed";
    EXPECT_FALSE(ack.success()) << "Insert should fail with empty vector";
}

TEST_P(MasterWorkerIntegrationTest, SearchWithMultipleWorkersNoInserts) {
    Query q;
    q.set_k(2);
    q.set_query_id(123);
    { static const float v[] = {1.0f, 2.0f, 3.0f, 4.0f}; set_float_bytes(q.mutable_vector_data(), v, 4); }

    AggregatedResult agg;
    grpc::ClientContext ctx;
    auto status = client_stub_->Search(&ctx, q, &agg);
    
    EXPECT_TRUE(status.ok()) << "Search failed: " << status.error_message();
    // EXPECT_EQ(agg.topk_size(), 0) << "Should not return more than k results";
    //MockAggregator should return k results even if no vectors are inserted
    
    // Verify results are sorted by distance 
    for (int i = 1; i < agg.topk_size(); ++i) {
        EXPECT_LE(agg.topk(i-1).distance(), agg.topk(i).distance()) 
            << "Results not sorted by distance";
    }
}

TEST_P(MasterWorkerIntegrationTest, SearchWithDifferentKValues) {
    std::vector<int> k_values = {1, 2, 3, 5};
    
    for (int k : k_values) {
        Query q;
        q.set_k(k);
        q.set_query_id(200 + k);
        { static const float v[] = {1.0f, 2.0f, 3.0f, 4.0f}; set_float_bytes(q.mutable_vector_data(), v, 4); }

        AggregatedResult agg;
        grpc::ClientContext ctx;
        auto status = client_stub_->Search(&ctx, q, &agg);

        EXPECT_TRUE(status.ok()) << "Search failed for k=" << k;
        EXPECT_LE(agg.topk_size(), k) << "Should not return more than k results";
    }
}

TEST_P(MasterWorkerIntegrationTest, ConcurrentSearches) {
    // tests whether multiple client requests can run in parallel
    const int num_concurrent = 5;
    std::vector<std::thread> threads;
    std::vector<bool> results(num_concurrent, false);

    // Pre-populate some vectors
    for (int id = 1000; id < 1020; ++id) {
        InsertData ins;
        ins.set_id(id);
        { static const float v[] = {1.0f, 2.0f, 3.0f, 4.0f}; set_float_bytes(ins.mutable_vector_data(), v, 4); }
        InsertAck ack;
        grpc::ClientContext ictx;
        auto s = client_stub_->Insert(&ictx, ins, &ack);
        ASSERT_TRUE(s.ok()) << "Pre-insert RPC failed";
    }
    
    for (int i = 0; i < num_concurrent; ++i) {
        threads.emplace_back([this, i, &results]() {
            Query q;
            q.set_k(2);
            q.set_query_id(400 + i);
            { static const float v[] = {1.0f, 2.0f, 3.0f, 4.0f}; set_float_bytes(q.mutable_vector_data(), v, 4); }

            AggregatedResult agg;
            grpc::ClientContext ctx;
            auto status = client_stub_->Search(&ctx, q, &agg);
            results[i] = status.ok() && agg.topk_size() > 0;
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    // TODO: Fix this test, remove MockQueryRouter and MockAggregator
    // for (int i = 0; i < num_concurrent; ++i) {
    //     EXPECT_TRUE(results[i]) << "Concurrent search " << i << " failed";
    // }
}

TEST_P(MasterWorkerIntegrationTest, ConcurrentSearchesAndInserts) {
    const int num_threads = 8;
    std::vector<std::thread> threads;
    std::vector<bool> ok(num_threads, false);

    // Launch mixed workload: even indices do searches, odd indices do inserts
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([this, i, &ok]() {
            if (i % 2 == 0) {
                // Search task
                Query q;
                q.set_k(2);
                q.set_query_id(600 + i);
                { static const float v[] = {1.0f, 2.0f, 3.0f, 4.0f}; set_float_bytes(q.mutable_vector_data(), v, 4); }
                AggregatedResult agg;
                grpc::ClientContext ctx;
                auto status = client_stub_->Search(&ctx, q, &agg);
                ok[i] = status.ok();
            } else {
                // Insert task
                InsertData ins;
                ins.set_id(2000 + i);
                { static const float v[] = {1.0f, 2.0f, 3.0f, 4.0f}; set_float_bytes(ins.mutable_vector_data(), v, 4); }
                InsertAck ack;
                grpc::ClientContext ctx;
                auto status = client_stub_->Insert(&ctx, ins, &ack);
                ok[i] = status.ok() && ack.success();
            }
        });
    }

    for (auto& t : threads) t.join();
    for (int i = 0; i < num_threads; ++i) EXPECT_TRUE(ok[i]);
}

INSTANTIATE_TEST_SUITE_P(NumWorkersCases,
                         MasterWorkerIntegrationTest,
                         ::testing::Values(1, 2, 5));

} // namespace ann

