"""Extract simple scalar events with stdlib; validate TFRecord CRC32C checksums."""
import argparse
import json
import struct
from pathlib import Path


def crc32c(data):
    crc=0xffffffff
    for b in data:
        crc ^= b
        for _ in range(8):crc=(crc>>1)^(0x82f63b78 if crc&1 else 0)
    return crc^0xffffffff


def masked_crc(data):
    crc=crc32c(data)
    return (((crc>>15)|(crc<<17))+0xa282ead8)&0xffffffff


def fields(data):
    def varint(pos):
        v=0
        for shift in range(0,70,7):
            if pos>=len(data):raise ValueError('truncated varint')
            byte=data[pos];pos+=1;v|=(byte&127)<<shift
            if not byte&128:return v,pos
        raise ValueError('oversized varint')
    i=0
    while i<len(data):
        key,i=varint(i);field,wire=key>>3,key&7
        if not field:raise ValueError('invalid protobuf field')
        if wire==0:value,i=varint(i)
        elif wire in (1,5):
            size=8 if wire==1 else 4
            if i+size>len(data):raise ValueError('truncated fixed field')
            value=data[i:i+size];i+=size
        elif wire==2:
            size,i=varint(i)
            if i+size>len(data):raise ValueError('truncated bytes field')
            value=data[i:i+size];i+=size
        else:raise ValueError('unsupported protobuf wire type')
        yield field,wire,value


def read_scalars(path,tags):
    result=[]
    with path.open('rb') as f:
        while length_bytes:=f.read(8):
            if len(length_bytes)!=8:raise ValueError('truncated TFRecord length')
            length_crc=f.read(4);length=struct.unpack('<Q',length_bytes)[0]
            if len(length_crc)!=4 or struct.unpack('<I',length_crc)[0]!=masked_crc(length_bytes):
                raise ValueError('invalid TFRecord length checksum')
            payload=f.read(length);payload_crc=f.read(4)
            if len(payload)!=length or len(payload_crc)!=4 or struct.unpack('<I',payload_crc)[0]!=masked_crc(payload):
                raise ValueError('invalid TFRecord payload checksum')
            event=list(fields(payload));step=next((v for k,w,v in event if k==2 and w==0),0)
            for k,w,summary in event:
                if k!=5 or w!=2:continue
                for sk,sw,value in fields(summary):
                    if sk!=1 or sw!=2:continue
                    vs=list(fields(value));tag=next((v.decode('utf8') for vk,vw,v in vs if vk==1 and vw==2),None)
                    scalar=next((struct.unpack('<f',v)[0] for vk,vw,v in vs if vk==2 and vw==5),None)
                    if tag in tags and scalar is not None:result.append({'step':step,'tag':tag,'value':scalar})
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('event',type=Path);p.add_argument('--tags',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out={'event':str(a.event),'records':read_scalars(a.event,set(a.tags)),'notes':'Only simple_value scalar events; TFRecord checksums verified.'}
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False),encoding='utf8');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
