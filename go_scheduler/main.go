package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"
)

type NodeStatus struct {
	Address   string    `json:"address"`
	Healthy   bool      `json:"healthy"`
	LastCheck time.Time `json:"last_check"`
	Message   string    `json:"message,omitempty"`
}

type Scheduler struct {
	mu            sync.RWMutex
	nodes         map[string]*NodeStatus
	elixirAddress string
}

func NewScheduler(addrs []string, elixirAddr string) *Scheduler {
	nodes := make(map[string]*NodeStatus)
	for _, a := range addrs {
		nodes[a] = &NodeStatus{
			Address:   a,
			Healthy:   true,
			LastCheck: time.Now(),
		}
	}
	return &Scheduler{
		nodes:         nodes,
		elixirAddress: elixirAddr,
	}
}

func (s *Scheduler) healthLoop(interval time.Duration) {
	client := http.Client{Timeout: 2 * time.Second}
	for {
		s.mu.RLock()
		addresses := make([]string, 0, len(s.nodes))
		for a := range s.nodes {
			addresses = append(addresses, a)
		}
		s.mu.RUnlock()

		var wg sync.WaitGroup
		for _, addr := range addresses {
			wg.Add(1)
			go func(a string) {
				defer wg.Done()
				// Query the node's /health endpoint
				resp, err := client.Get(a + "/health")
				healthy := err == nil && resp.StatusCode == 200
				
				var msg string
				if err != nil {
					msg = err.Error()
				} else {
					resp.Body.Close()
				}

				s.mu.Lock()
				status := s.nodes[a]
				wasHealthy := status.Healthy
				status.Healthy = healthy
				status.LastCheck = time.Now()
				status.Message = msg
				s.mu.Unlock()

				if wasHealthy && !healthy {
					log.Printf("[WARNING] Node %s is down! Notifying Elixir supervisor...", a)
					s.notifySupervisor(a)
				}
			}(addr)
		}
		wg.Wait()
		time.Sleep(interval)
	}
}

func (s *Scheduler) notifySupervisor(nodeAddr string) {
	// Call Elixir supervisor's crash handler endpoint
	url := fmt.Sprintf("%s/node_down?addr=%s", s.elixirAddress, nodeAddr)
	resp, err := http.Post(url, "application/json", nil)
	if err != nil {
		log.Printf("[ERROR] Failed to notify Elixir supervisor at %s: %v", url, err)
		return
	}
	resp.Body.Close()
	log.Printf("[INFO] Successfully notified Elixir supervisor about crash of node %s", nodeAddr)
}

func (s *Scheduler) statusHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	s.mu.RLock()
	defer s.mu.RUnlock()
	json.NewEncoder(w).Encode(s.nodes)
}

func (s *Scheduler) healthHandler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

func main() {
	port := flag.Int("port", 9000, "Port to run the scheduler server on")
	elixirAddr := flag.String("elixir", "http://localhost:4000", "Elixir supervisor router address")
	flag.Parse()

	// Initial set of nodes to monitor (usually defined in config)
	nodeAddrs := []string{
		"http://localhost:8001", // Mock Node 1
		"http://localhost:8002", // Mock Node 2
	}

	scheduler := NewScheduler(nodeAddrs, *elixirAddr)
	
	// Start health checking background thread (checks every 5 seconds)
	go scheduler.healthLoop(5 * time.Second)

	http.HandleFunc("/status", scheduler.statusHandler)
	http.HandleFunc("/health", scheduler.healthHandler)

	addr := fmt.Sprintf(":%d", *port)
	log.Printf("[INFO] Go Scheduler listening on port %d...", *port)
	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatalf("Server startup failed: %v", err)
	}
}
