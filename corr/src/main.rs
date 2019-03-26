
use std::env;
use std::io::{BufRead, BufReader};
use std::fs::File;
use std::collections::VecDeque;

fn sum_a_file(filename:&String, corr_len:usize) -> std::io::Result<i32> {
    let file = File::open(filename).unwrap();

    let mut sum : i128 = 0;
    let mut buf : VecDeque<i128> = VecDeque::new();
    for _ in 1..corr_len {
        buf.push_front(0);
    }
    let mut corr = vec![0; corr_len];
    
    let mut reader = BufReader::new(file);
    let mut line = String::new();

    let mut counter :i128 = 0;
    let mut iter:i128 = -1;
    let mut len = 1;
    let mut errors = 0;

    while len>0{
        // Read line and pop last \r\n characters
        iter +=1;
        len = reader.read_line(&mut line)?;
        line.pop();
        line.pop();

        // Parse the integer from line
        match line.parse::<i32>(){
            Err(_)=> errors+=1,
            
            Ok(n)=> {
                let mut n = n as i128;
                if n>10000{
                    continue
                }
                if n==12{
                    n=0;
                }else{
                    n=1;
                }
                counter +=1;
                if iter%2==0{
                    n = -n+1;
                }
                sum += n;
                buf.push_front(n);
                for (i, x) in buf.iter().enumerate() {
                    corr[i] += n*x;
                }
                buf.pop_back();
            }
        }
        line.clear();

        if counter%10000000==0{
            println!("count {}", counter);
        }
        if counter==300000000{
            println!("Maximum length of 1M riched.");
        }
    }
    println!("Error count: {}",errors);

    println!("Data sum: {}", sum);
    println!("Data count: {}",counter);

    let mean = ( sum as f64)/(counter as f64);
    println!("Mean: {}",mean);
    let mut res = Vec::new();
    for x in corr.iter() {
        println!("Corr raw: {}", *x);
        let n =  counter as f64;
        let centered  = ( *x*counter - sum*sum) as f64 / n;

        println!("Corr center: {}", centered);
        println!("Corr: {}\n",centered/n);
        res.push(centered/n);

    }
    println!("\nResult: {:?}",res);
    return Ok(1);
}
fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1) {
        None => println!("Please provide a filename"),
        Some(f)=> {
            let ten = String::from("10");
            let l = args.get(2).unwrap_or(&ten);
            let cl  =l.parse::<usize>().unwrap();
            println!("Reading file {}", f);
            let res = sum_a_file(f,cl);
            match res{
                Ok(_) => println!("Done."),
                Err(_) => println!("Exited with error")
            }
        }
    }
}
