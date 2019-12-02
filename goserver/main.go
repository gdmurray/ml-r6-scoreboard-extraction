package main

import (
	"fmt"
    "log"
	"net/http"
	//"io/ioutil"
    "strconv"
	"encoding/json"
	"os"
	"github.com/gorilla/mux"
	"github.com/gomodule/redigo/redis"
	tg "github.com/galeone/tfgo"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	ts "github.com/otiai10/gosseract"
	//"encoding/binary"
	"gocv.io/x/gocv"
	//"bytes"
    "strings"
    //"image/jpeg"
	"encoding/base64"
	//"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

const (
	graphFile  = "/models/model/model.pb"
	// labelsFile = "/model/imagenet_comp_graph_label_strings.txt"
)

var client redis.Conn
var model *tg.Model

func parsePredictArray(arr []uint8) [1][50][150][1]float32 {
	counter := 0
	var imgArr [1][50][150][1]float32
	for i := 0; i < 50; i++ {
		for j := 0; j < 150; j++{
			imgArr[0][i][j][0] = float32(arr[counter])
			//fmt.Println("Iter vals: ", i,j, counter)
			counter++
		}
	}
	return imgArr
}

func InitRedisClient() redis.Conn{
    conn, err := redis.Dial("tcp", "redis:6379")
    if err != nil {
        fmt.Println("Couldnt connect to redis")
    }
    pong, err := conn.Do("PING")
    if err != nil {
       fmt.Println("Couldnt Ping Redis Server")
    }
    fmt.Sprintf("Got a response from Server %s", pong)
    client = conn
    return client
}

func OCRHandler(w http.ResponseWriter,r *http.Request){
	fmt.Println("Received a ping.")
    params := mux.Vars(r)
    img_id := params["image_id"]
    score_n := params["score_num"]
	text_n := params["text_num"]
	type_key, _ := r.URL.Query()["type"]
    row_key, _ := r.URL.Query()["x"]
	col_key, _ := r.URL.Query()["y"]
	text_type := type_key[0]

    rows,_ := strconv.Atoi(row_key[0])
	cols,_ := strconv.Atoi(col_key[0])

	text1_key, _ := r.URL.Query()["text1"]
	text2_key, _ := r.URL.Query()["text2"]
	text3_key, _ := r.URL.Query()["text3"]
	text4_key, _ := r.URL.Query()["text4"]
	text1 := text1_key[0]
	text2 := text2_key[0]
	text3 := text3_key[0]
	text4 := text4_key[0]
	text_map := map[string]string{"text1": text1, "text2": text2, "text3": text3, "text4": text4}
    fmt.Println(text_map)
	fmt.Println("Got Type: ", text_type)
    fmt.Println("Got dims: ", cols, rows)
	fmt.Println(img_id, score_n, text_n)

	tessClient := ts.NewClient()
	defer client.Close()

	if text_type == "text"{
		tessClient.SetPageSegMode(12);
	}else if text_type == "number"{
		tessClient.SetPageSegMode(10);
		tessClient.SetVariable("tessedit_ocr_engine_mode", "3");
	}

	texts :=[4]string{"text1", "text2", "text3", "text4"}

	for _, text := range texts {
        b64_string := text_map[text]
		fmt.Printf("b64 before: %S\n", b64_string)
		b64_string = strings.Replace(b64_string, "b'", "", -1)
		b64_string = strings.Replace(b64_string, "'", "", -1)
		fmt.Printf("b64 After: %T %S\n", b64_string, b64_string)
		img_data, _ := base64.RawURLEncoding.DecodeString(b64_string)
		mat, _ := gocv.NewMatFromBytes(rows, cols, gocv.MatTypeCV8S, img_data)
        tessClient.SetImageFromBytes(mat.ToBytes())
        boxes, _ := tessClient.GetBoundingBoxes(ts.RIL_BLOCK)
        fmt.Println(boxes)
	}
	json.NewEncoder(w).Encode(map[string]string{"status": "OK"})
}

func PredictHandler(w http.ResponseWriter, r *http.Request){
    fmt.Println("Received a predict.")
    params := mux.Vars(r)
    img_id := params["image_id"]
    score_n := params["score_num"]
    text_n := params["text_num"]
    bytes_val, err := redis.Bytes(client.Do("GET", fmt.Sprintf("%s/%s/%s/%s", img_id,score_n, text_n, "predict")))

    if err != nil {
        panic(err)
    }
	fmt.Println("Image is of length: ", len(bytes_val))
	if len(bytes_val) != 7500{
		panic("Image is not proper size")
	}
	imgArr := parsePredictArray(bytes_val)

	xInput, _ := tf.NewTensor(imgArr)

	results := model.Exec([]tf.Output{
		model.Op("linear/head/predictions/probabilities/Softmax", 0),
	}, map[tf.Output]*tf.Tensor{
		model.Op("input_1", 0): xInput,
	})

	predictions := results[0].Value().([][]float32)

	fmt.Println(predictions)
	json.NewEncoder(w).Encode(map[string]string{"status": "OK"})

}

func main(){
	fmt.Println("In Tensorflow Version: ", tf.Version())
	os.Setenv("TF_CPP_MIN_LOG_LEVEL", "2")
	fmt.Println("Starting Go Server Server...")
	InitRedisClient()

	r := mux.NewRouter()
	r.HandleFunc("/ocr/{image_id}/{score_num}/{text_num}", OCRHandler).Methods("GET")
	r.HandleFunc("/predict/{image_id}/{score_num}/{text_num}", PredictHandler).Methods("GET")
	//fmt.Println(os.File.Readdir("/models/model"))
	model = tg.LoadModel("/models/model", []string{"serve"}, nil)
	//graph, err := loadModel()
	//if err != nil{
	//	panic(err)
	//}
	fmt.Println(model)
	fmt.Println("Listening on port :8080 . . .")
    log.Fatal(http.ListenAndServe(":8080", r))
}