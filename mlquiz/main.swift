//
//  main.swift
//  mlquiz
//
//  Created by zhongdian on 2020/11/10.
//

import Foundation
import Numerics
import SwiftCSV

// expect x[0] always equals 1.0
func hypothesis<T: Real>(thetas: [T], x: [T]) -> T {
    let index: T = -thetas[0] - thetas[1] * x[1] - thetas[2] * x[2]
    return 1 / ( 1 + .exp(index))
}

func singleCostMultiX<T: Real>(thetas: [T], x: [T], y: T, index: Int) -> T {
    let hypo = hypothesis(thetas: thetas, x: x);
    let cost = hypo - y;
    return cost * x[index]
}

func gradientDecent<T: Real>(thetas: [T], alpha: T, array: Array<Array<T>>) -> [T] {
    var currentSum0: T = T(0);
    var currentSum1: T = T(0);
    var currentSum2: T = T(0);
    for row in array {
        let x: [T] = [T(1), row[1], row[2]];
        let y = row[3]
        currentSum0 += singleCostMultiX(thetas: thetas, x: x, y: y, index: 0)
        currentSum1 += singleCostMultiX(thetas: thetas, x: x, y: y, index: 1)
        currentSum2 += singleCostMultiX(thetas: thetas, x: x, y: y, index: 2)
    }
    let theta0 = thetas[0] - alpha * currentSum0 /  T(array.count);
    let theta1 = thetas[1] - alpha * currentSum1 /  T(array.count);
    let theta2 = thetas[2] - alpha * currentSum2 /  T(array.count);
    
    return [theta0, theta1, theta2];
}

func test<T: Real>(thetas: [T], array: Array<Array<T>>) -> T {
    var sum = T(0);
    for row in array {
        let x: [T] = [T(1), row[1], row[2]];
        let y: T = row[3];
        let hypo = hypothesis(thetas: thetas, x: x);
        let diff = T(0) - y * .log(hypo) - (T(1) - y) * .log(T(1) - hypo);
        sum += diff * diff;
    }
    return sum;
}

func LogisticRegression() {
    let alpha  = 0.0000001;
    var thetas = [0.0, 0.0, 0.0];

    let trainData = try! CSV(url: URL(fileURLWithPath: "/Users/desmond/Desktop/learn/mlquiz/mlquiz/quiz/breast-cancer-train.csv"));
    var trainArray = Array<Array<Double>>()
    for row in trainData.enumeratedRows {
        let item = Array(arrayLiteral: Double(row[0])!, Double(row[1])!, Double(row[2])!, Double(row[3])!)
        trainArray.append(item)
    }
    
    let testData = try! CSV(url: URL(fileURLWithPath: "/Users/desmond/Desktop/learn/mlquiz/mlquiz/quiz/breast-cancer-test.csv"));
    var testArray = Array<Array<Double>>()
    for row in testData.enumeratedRows {
        let item = Array(arrayLiteral: Double(row[0])!, Double(row[1])!, Double(row[2])!, Double(row[3])!)
        testArray.append(item)
    }
    
    var iterate = 0;
    while true {
        iterate += 1;
        thetas = gradientDecent(thetas: thetas, alpha: alpha, array: trainArray);
        if iterate % 100 == 0 {
            print(thetas);
            let sum = test(thetas: thetas, array: testArray);
            print(sum);
            print("======================")
        }
    }
}

LogisticRegression()
