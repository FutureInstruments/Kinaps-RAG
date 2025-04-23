@description('The name of the DNS zone to be created.  Must have at least 2 segments, e.g. hostname.org')
param dnsRecordZone string

@description('The name of the DNS record to be created.  The name is relative to the zone, not the FQDN.')
param dnsRecordName string

param dnsCNAME string
param dnsTXT string[]

resource zone 'Microsoft.Network/dnsZones@2018-05-01' existing = {
  name: dnsRecordZone
}


resource record 'Microsoft.Network/dnsZones/CNAME@2023-07-01-preview' = {
  parent: zone
  name: dnsRecordName
  properties: {
    TTL: 300
    CNAMERecord: {
      cname: dnsCNAME
    }
  }
}

resource verification 'Microsoft.Network/dnsZones/TXT@2023-07-01-preview' = {
  parent: zone
  name: 'asuid.${dnsRecordName}'
  properties: {
    TTL: 300
    TXTRecords: [
      {
        value: dnsTXT
      }
    ]
  }
}
