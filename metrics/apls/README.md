# APLS Usage
This APLS metric is implemented in GOLANG. To run it, we first need to convert the graph files into json format. 
```bash
python convert.py example/gt.p example/gt.json
python convert.py example/prop.p example/prop.json
```
Then we can use main.go to get the APLS metric.
```bash
go main.go example/gt.json example/prop.json aplsresult.txt 
```

The parameters in this APLS implementation is configed for 2048x2048 tiles. To change them, please see line 15-25 in main.go.  